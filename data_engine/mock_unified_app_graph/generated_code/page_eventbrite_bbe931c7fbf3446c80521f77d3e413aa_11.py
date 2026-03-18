# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_11
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13.png
# step_index: 11/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill=(250, 248, 252))  # very light off-white background

# Top status bar area (~50-96px)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill=(245, 246, 248))  # subtle light status bar background
# thin bottom divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(230, 228, 235), width=1)

# Hero/image area background (under the image that will be pasted)
hero_top = status_h
hero_bottom = 560
draw.rectangle((0, hero_top, 1440, hero_bottom), fill=(28, 28, 28))  # dark backdrop for the hero image area
# soft lighter overlay bar near bottom of hero to suggest fade
overlay_y = hero_bottom - 40
draw.rectangle((0, overlay_y, 1440, hero_bottom), fill=(40, 40, 40))

# Divider under hero
draw.line((48, hero_bottom + 6, 1440 - 48, hero_bottom + 6), fill=(230, 228, 235), width=2)

# Organizer / follow card (rounded rectangle behind avatar + follow button)
card_x1, card_x2 = 48, 1392
card_y1, card_y2 = 1108, 1288  # sits approximately around the detected organizer area
card_radius = 28
draw.rounded_rectangle((card_x1, card_y1, card_x2, card_y2), radius=card_radius, fill=(250, 250, 252), outline=(235, 232, 240), width=1)

# Subtle inset shadow line under the card
draw.line((card_x1 + 8, card_y2, card_x2 - 8, card_y2), fill=(238, 236, 243), width=1)

# Section separators between content blocks
separators = [1408, 1536, 1872, 2140]  # y positions for horizontal rules
for y in separators:
    draw.line((48, y, 1392, y), fill=(235, 233, 239), width=2)

# "About this event" tag pill background (rounded)
pill_x1, pill_y1 = 48, 2032
pill_w, pill_h = 600, 68
pill_radius = 34
draw.rounded_rectangle((pill_x1, pill_y1, pill_x1 + pill_w, pill_y1 + pill_h), radius=pill_radius, fill=(246, 246, 249), outline=(235, 233, 239), width=1)

# Light content block background for the "Location" / map area near bottom
loc_block_x1, loc_block_y1 = 48, 2508
loc_block_x2, loc_block_y2 = 1392, 2860
draw.rectangle((loc_block_x1, loc_block_y1, loc_block_x2, loc_block_y2), fill=(255, 255, 255))
# subtle border on top of location block
draw.line((loc_block_x1, loc_block_y1, loc_block_x2, loc_block_y1), fill=(235, 233, 239), width=2)

# Large subtle divider above "About this event" heading
about_div_y = 1888
draw.line((48, about_div_y, 1392, about_div_y), fill=(236, 234, 241), width=1)

# Bottom safe area faint shading
draw.rectangle((0, 2900, 1440, 2960), fill=(248, 247, 250))

# Light vertical guides/padding markers (non-intrusive, very faint) to reflect layout gutters
gutter_color = (250, 249, 251)
draw.line((48, 0, 48, 2960), fill=gutter_color, width=1)
draw.line((1392, 0, 1392, 2960), fill=gutter_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1163), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/03_icon_9.13.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 69)
    canvas.paste(_c4, (1154, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [1154, 2, 1203, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 46, 63)
    canvas.paste(_c5, (1326, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [1326, 3, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 101, 64)
    canvas.paste(_c6, (1213, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 2, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/07_icon_Los_Angeles_Convention_Center_1201_South.png
try:
    _c7 = get_crop(7, 226, 144)
    canvas.paste(_c7, (1166, 2518), _c7)
except Exception:
    pass
layout["Los_Angeles_Convention_Ce"] = [1166, 2518, 1392, 2662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/08_icon_I_00_PM.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1116, 108), _c8)
except Exception:
    pass
layout["I:00_PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/09_icon_9.13.png
try:
    _c9 = get_crop(9, 52, 60)
    canvas.paste(_c9, (117, 3), _c9)
except Exception:
    pass
layout["9.13"] = [117, 3, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 57)
    canvas.paste(_c10, (183, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [183, 4, 234, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 51, 57)
    canvas.paste(_c11, (316, 6), _c11)
except Exception:
    pass
layout["icon_11"] = [316, 6, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 55, 58)
    canvas.paste(_c12, (247, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 4, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/13_text_9.13.png
try:
    _c13 = get_crop(13, 91, 43)
    canvas.paste(_c13, (20, 17), _c13)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/14_text_Saturday_April_13.png
try:
    _c14 = get_crop(14, 453, 77)
    canvas.paste(_c14, (38, 758), _c14)
except Exception:
    pass
layout["Saturday;_April_13"] = [38, 758, 491, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/15_text_I_00_PM.png
try:
    _c15 = get_crop(15, 207, 56)
    canvas.paste(_c15, (523, 766), _c15)
except Exception:
    pass
layout["I:00_PM"] = [523, 766, 730, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/16_text_Minorities_Building_Wealth_with.png
try:
    _c16 = get_crop(16, 269, 144)
    canvas.paste(_c16, (288, 1123), _c16)
except Exception:
    pass
layout["Minorities_Building_Wealt"] = [288, 1123, 557, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/17_text_At_the_Franchise_Expo_West.png
try:
    _c17 = get_crop(17, 331, 144)
    canvas.paste(_c17, (1013, 1163), _c17)
except Exception:
    pass
layout["At_the_Franchise_Expo_Wes"] = [1013, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/18_text_Sociallybuzz.png
try:
    _c18 = get_crop(18, 269, 144)
    canvas.paste(_c18, (288, 1123), _c18)
except Exception:
    pass
layout["Sociallybuzz"] = [288, 1123, 557, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/19_text_129_Followers.png
try:
    _c19 = get_crop(19, 269, 144)
    canvas.paste(_c19, (288, 1123), _c19)
except Exception:
    pass
layout["129_Followers"] = [288, 1123, 557, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/20_text_Los_Angeles_Convention_Center.png
try:
    _c20 = get_crop(20, 1344, 144)
    canvas.paste(_c20, (48, 1390), _c20)
except Exception:
    pass
layout["Los_Angeles_Convention_Ce"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/21_text_hrs.png
try:
    _c21 = get_crop(21, 77, 50)
    canvas.paste(_c21, (176, 1547), _c21)
except Exception:
    pass
layout["hrs"] = [176, 1547, 253, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/22_text_Refund_policy.png
try:
    _c22 = get_crop(22, 299, 63)
    canvas.paste(_c22, (138, 1653), _c22)
except Exception:
    pass
layout["Refund_policy"] = [138, 1653, 437, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/23_text_The_organizer_will_review_refund_request.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1390), _c23)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/24_text_About_this_event.png
try:
    _c24 = get_crop(24, 452, 61)
    canvas.paste(_c24, (45, 1953), _c24)
except Exception:
    pass
layout["About_this_event"] = [45, 1953, 497, 2014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/25_text_Business_Professional_._Startups_Small_B.png
try:
    _c25 = get_crop(25, 234, 144)
    canvas.paste(_c25, (48, 2300), _c25)
except Exception:
    pass
layout["Business_&_Professional_."] = [48, 2300, 282, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/26_text_We_invite_you_to_a_value-packed_educatio.png
try:
    _c26 = get_crop(26, 234, 144)
    canvas.paste(_c26, (48, 2300), _c26)
except Exception:
    pass
layout["We_invite_you_to_a_value-"] = [48, 2300, 282, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/27_text_Read_more.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 2300), _c27)
except Exception:
    pass
layout["Read_more"] = [48, 2300, 282, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/28_text_Location.png
try:
    _c28 = get_crop(28, 246, 61)
    canvas.paste(_c28, (41, 2564), _c28)
except Exception:
    pass
layout["Location"] = [41, 2564, 287, 2625]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/29_text_Show_map.png
try:
    _c29 = get_crop(29, 226, 144)
    canvas.paste(_c29, (1166, 2518), _c29)
except Exception:
    pass
layout["Show_map"] = [1166, 2518, 1392, 2662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_11_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-13/30_clickable_Organizer_profile_picture.png
try:
    _c30 = get_crop(30, 144, 144)
    canvas.paste(_c30, (96, 1162), _c30)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1162, 240, 1306]
