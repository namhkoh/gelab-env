# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_19
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21.png
# step_index: 19/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall page background
draw.rectangle([(0, 0), (1440, 2960)], fill=(243, 245, 248))  # soft off-white page background

# Status bar (top area ~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(190, 193, 196))  # subtle grey status bar

# Header / toolbar background under status bar (~72 -> 168)
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))  # white header
# header bottom divider
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(225, 227, 230), width=2)

# Large content region background (a subtle panel behind content)
content_top = header_bottom + 16
content_left = 48
content_right = 1392
# full content subtle card area - do not overlap detected individual image areas (we'll draw separate card backgrounds)
draw.rectangle([(0, content_top), (1440, 2320)], fill=(243, 245, 248))  # keep page background continuation

# Video/card background 1 (behind detected image at (58,581) size 1323x755)
v1_x = 58
v1_y = 581
v1_w = 1323
v1_h = 755
v1_rect = [ (v1_x-6, v1_y-10), (v1_x + v1_w + 6, v1_y + v1_h + 10) ]
# subtle shadow
draw.rounded_rectangle(v1_rect, radius=14, fill=(236, 238, 241))
# inner white card behind the video image
draw.rounded_rectangle([ (v1_x, v1_y), (v1_x + v1_w, v1_y + v1_h) ], radius=10, fill=(255, 255, 255), outline=(220, 222, 225), width=2)

# Divider between first and second video areas
divider_y = v1_y + v1_h + 18
draw.line([(72, divider_y), (1368, divider_y)], fill=(225, 227, 230), width=1)

# Video/card background 2 (behind detected image at (58,1357) size 1323x755)
v2_x = 58
v2_y = 1357
v2_w = 1323
v2_h = 755
v2_rect = [ (v2_x-6, v2_y-10), (v2_x + v2_w + 6, v2_y + v2_h + 10) ]
draw.rounded_rectangle(v2_rect, radius=14, fill=(236, 238, 241))
draw.rounded_rectangle([ (v2_x, v2_y), (v2_x + v2_w, v2_y + v2_h) ], radius=10, fill=(255,255,255), outline=(220,222,225), width=2)

# Horizontal separator above "Read less" area (approx near bottom of second video)
sep_y = v2_y + v2_h + 26
draw.line([(72, sep_y), (1368, sep_y)], fill=(225,227,230), width=1)

# Ticket selection card background (rounded rectangle with blue border)
ticket_left = 72
ticket_right = 1368
ticket_top = 2360
ticket_bottom = 2680
# subtle outer shadow
draw.rounded_rectangle([ (ticket_left-8, ticket_top+6), (ticket_right+8, ticket_bottom+10) ], radius=20, fill=(237,239,244))
# white ticket card
draw.rounded_rectangle([ (ticket_left, ticket_top), (ticket_right, ticket_bottom) ], radius=18, fill=(255,255,255), outline=(59, 91, 248), width=6)

# thin inner divider line inside ticket card (to suggest separation of title and price)
inner_div_y = ticket_top + 86
draw.line([(ticket_left+28, inner_div_y), (ticket_right-28, inner_div_y)], fill=(234,236,241), width=2)

# subtle separator above the ticket card to separate content sections
draw.line([(48, ticket_top-28), (1392, ticket_top-28)], fill=(225,227,230), width=1)

# Bottom area (reserve button area background) - keep neutral and minimal, do not duplicate the actual button
bottom_area_top = ticket_bottom + 22
draw.rectangle([(0, bottom_area_top), (1440, 2960)], fill=(243,245,248))

# Final subtle page right/left margins - faint vertical guides for consistent look (very light)
draw.line([(48, header_bottom+6), (48, 2960)], fill=(251,251,252), width=1)
draw.line([(1392, header_bottom+6), (1392, 2960)], fill=(251,251,252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/00_icon_Socot.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["Socot"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/02_icon_Empowering_Diversity_in_Franchising_How_.png
try:
    _c2 = get_crop(2, 1289, 20)
    canvas.paste(_c2, (75, 486), _c2)
except Exception:
    pass
layout["Empowering_Diversity_in_F"] = [75, 486, 1364, 506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1296, 132)
    canvas.paste(_c3, (72, 2756), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/04_icon_Jalbuzz.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Jalbuzz"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 92, 103)
    canvas.paste(_c6, (1108, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 57)
    canvas.paste(_c7, (248, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 5, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 58)
    canvas.paste(_c8, (316, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [316, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/09_icon_9.14.png
try:
    _c9 = get_crop(9, 51, 58)
    canvas.paste(_c9, (184, 3), _c9)
except Exception:
    pass
layout["9.14"] = [184, 3, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/10_icon_Ae.png
try:
    _c10 = get_crop(10, 1323, 755)
    canvas.paste(_c10, (58, 581), _c10)
except Exception:
    pass
layout["Ae"] = [58, 581, 1381, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 46, 64)
    canvas.paste(_c11, (1156, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1156, 2, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/12_icon_9.14.png
try:
    _c12 = get_crop(12, 52, 59)
    canvas.paste(_c12, (117, 3), _c12)
except Exception:
    pass
layout["9.14"] = [117, 3, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 57)
    canvas.paste(_c13, (1327, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1327, 5, 1372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/14_icon_9.14.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (36, 108), _c14)
except Exception:
    pass
layout["9.14"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/15_icon_Free.png
try:
    _c15 = get_crop(15, 134, 103)
    canvas.paste(_c15, (98, 2573), _c15)
except Exception:
    pass
layout["Free"] = [98, 2573, 232, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 58, 60)
    canvas.paste(_c16, (1213, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [1213, 3, 1271, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 48, 62)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [383, 2, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 43, 59)
    canvas.paste(_c18, (1270, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1270, 3, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/19_icon_Franchising_Learn_the_7_Habits_of_Highly.png
try:
    _c19 = get_crop(19, 1323, 755)
    canvas.paste(_c19, (58, 1357), _c19)
except Exception:
    pass
layout["Franchising:_Learn_the_7_"] = [58, 1357, 1381, 2112]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/20_icon_Free.png
try:
    _c20 = get_crop(20, 75, 72)
    canvas.paste(_c20, (249, 2588), _c20)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/21_icon_Share.png
try:
    _c21 = get_crop(21, 65, 84)
    canvas.paste(_c21, (1285, 581), _c21)
except Exception:
    pass
layout["Share"] = [1285, 581, 1350, 665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/22_icon_eadthfsl.png
try:
    _c22 = get_crop(22, 66, 66)
    canvas.paste(_c22, (75, 494), _c22)
except Exception:
    pass
layout["(eadthfsl"] = [75, 494, 141, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/23_text_9.14.png
try:
    _c23 = get_crop(23, 94, 41)
    canvas.paste(_c23, (20, 17), _c23)
except Exception:
    pass
layout["9.14"] = [20, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/24_text_Minorities_Building-.png
try:
    _c24 = get_crop(24, 1289, 20)
    canvas.paste(_c24, (75, 486), _c24)
except Exception:
    pass
layout["Minorities_Building-"] = [75, 486, 1364, 506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/25_text_Read_less.png
try:
    _c25 = get_crop(25, 206, 144)
    canvas.paste(_c25, (48, 2131), _c25)
except Exception:
    pass
layout["Read_less"] = [48, 2131, 254, 2275]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/26_text_Complimentary_Access.png
try:
    _c26 = get_crop(26, 75, 72)
    canvas.paste(_c26, (249, 2588), _c26)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/27_clickable_Mute.png
try:
    _c27 = get_crop(27, 66, 66)
    canvas.paste(_c27, (141, 494), _c27)
except Exception:
    pass
layout["Mute"] = [141, 494, 207, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/28_clickable_0_00_1_09.png
try:
    _c28 = get_crop(28, 100, 66)
    canvas.paste(_c28, (207, 494), _c28)
except Exception:
    pass
layout["0:00___1:09"] = [207, 494, 307, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/29_clickable_Subtitles_closed_captions.png
try:
    _c29 = get_crop(29, 66, 66)
    canvas.paste(_c29, (1075, 494), _c29)
except Exception:
    pass
layout["Subtitles_closed_captions"] = [1075, 494, 1141, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/30_clickable_Settings.png
try:
    _c30 = get_crop(30, 65, 66)
    canvas.paste(_c30, (1141, 494), _c30)
except Exception:
    pass
layout["Settings"] = [1141, 494, 1206, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/31_clickable_Watch_on_YouTube.png
try:
    _c31 = get_crop(31, 92, 66)
    canvas.paste(_c31, (1206, 494), _c31)
except Exception:
    pass
layout["Watch_on_YouTube"] = [1206, 494, 1298, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/32_clickable_Full_screen.png
try:
    _c32 = get_crop(32, 66, 66)
    canvas.paste(_c32, (1298, 494), _c32)
except Exception:
    pass
layout["Full_screen"] = [1298, 494, 1364, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/33_clickable_Photo_image_of_Sociallybuzz_Inc.png
try:
    _c33 = get_crop(33, 66, 66)
    canvas.paste(_c33, (68, 591), _c33)
except Exception:
    pass
layout["Photo_image_of_Sociallybu"] = [68, 591, 134, 657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/34_clickable_Empowering_Diversity_in_Franchising_How_.png
try:
    _c34 = get_crop(34, 1127, 33)
    canvas.paste(_c34, (144, 610), _c34)
except Exception:
    pass
layout["Empowering_Diversity_in_F"] = [144, 610, 1271, 643]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/35_clickable_Play.png
try:
    _c35 = get_crop(35, 93, 66)
    canvas.paste(_c35, (673, 926), _c35)
except Exception:
    pass
layout["Play"] = [673, 926, 766, 992]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/36_clickable_Watch_on_YouTube.png
try:
    _c36 = get_crop(36, 238, 65)
    canvas.paste(_c36, (58, 1264), _c36)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 1264, 296, 1329]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/37_clickable_Share.png
try:
    _c37 = get_crop(37, 65, 84)
    canvas.paste(_c37, (1285, 1357), _c37)
except Exception:
    pass
layout["Share"] = [1285, 1357, 1350, 1441]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/38_clickable_Photo_image_of_Sociallybuzz_Inc.png
try:
    _c38 = get_crop(38, 66, 66)
    canvas.paste(_c38, (68, 1367), _c38)
except Exception:
    pass
layout["Photo_image_of_Sociallybu"] = [68, 1367, 134, 1433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/39_clickable_Minorities_in_Franchising_Learn_the_7_Ha.png
try:
    _c39 = get_crop(39, 1127, 33)
    canvas.paste(_c39, (144, 1386), _c39)
except Exception:
    pass
layout["Minorities_in_Franchising"] = [144, 1386, 1271, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/40_clickable_Play.png
try:
    _c40 = get_crop(40, 93, 66)
    canvas.paste(_c40, (673, 1702), _c40)
except Exception:
    pass
layout["Play"] = [673, 1702, 766, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_19_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-21/41_clickable_Watch_on_YouTube.png
try:
    _c41 = get_crop(41, 238, 65)
    canvas.paste(_c41, (58, 2040), _c41)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 2040, 296, 2105]
