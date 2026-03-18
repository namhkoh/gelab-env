# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_07
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9.png
# step_index: 7/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar background
draw.rectangle([(0, 0), (1440, 86)], fill="#b9c1bd")  # muted gray-green status bar
draw.line([(0, 86), (1440, 86)], fill="#a9b1ad", width=1)

# Top snackbar / confirmation banner (under status bar)
draw.rectangle([(0, 86), (1440, 170)], fill="#e9f4ee")  # pale mint banner
draw.line([(0, 170), (1440, 170)], fill="#d8e6df", width=1)

# Hero image background area (under the banner) - large visual band
draw.rectangle([(0, 170), (1440, 460)], fill="#69a7b6")  # teal/blue band (image placeholder background)
# subtle top and bottom separators for the hero
draw.line([(0, 170), (1440, 170)], fill="#5c97a6", width=1)
draw.line([(0, 460), (1440, 460)], fill="#5c97a6", width=1)

# Page base background (ensure consistent neutral white)
draw.rectangle([(0, 460), (1440, 2960)], fill="#ffffff")

# Organizer card background with subtle shadow
card_x0, card_y0 = 40, 1050
card_x1, card_y1 = 1400, 1240
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    [(card_x0 + shadow_offset, card_y0 + shadow_offset),
     (card_x1 + shadow_offset, card_y1 + shadow_offset)],
    radius=28, fill="#e9e7ea"
)
# card fill
draw.rounded_rectangle(
    [(card_x0, card_y0), (card_x1, card_y1)],
    radius=28, fill="#f6f5f8"
)
# subtle inner divider on the card (light)
draw.line([(card_x0 + 24, card_y0 + 92), (card_x1 - 24, card_y0 + 92)], fill="#e1dfe3", width=1)

# Section separators between content blocks
sep_x0, sep_x1 = 40, 1400
separators_y = [1400, 1760, 1960, 2120]
for y in separators_y:
    draw.line([(sep_x0, y), (sep_x1, y)], fill="#efedf0", width=2)

# "About this event" area background hint (subtle)
about_area = (40, 1860, 1400, 2000)
draw.rectangle([about_area[0], about_area[1], about_area[2], about_area[3]], fill="#ffffff")
# light pill background for category tag (no text)
draw.rounded_rectangle([(48, 1910), (420, 1970)], radius=32, fill="#eef3f6")

# Ticket selection card (outlined rounded rectangle)
ticket_x0, ticket_y0 = 40, 2160
ticket_x1, ticket_y1 = 1400, 2360
# border shadow
draw.rounded_rectangle(
    [(ticket_x0 + 4, ticket_y0 + 6), (ticket_x1 + 4, ticket_y1 + 6)],
    radius=18, fill="#e9e9ec"
)
# main ticket card with blue outline
draw.rounded_rectangle(
    [(ticket_x0, ticket_y0), (ticket_x1, ticket_y1)],
    radius=18, fill="#ffffff", outline="#3658e0", width=6
)

# Horizontal rule above bottom action area (to separate from reserve button)
draw.line([(40, 2720), (1400, 2720)], fill="#efeef1", width=2)

# Subtle bottom safe-area background (below reserve button zone)
draw.rectangle([(0, 2760), (1440, 2960)], fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/01_icon_Going_fast.png
try:
    _c1 = get_crop(1, 335, 85)
    canvas.paste(_c1, (41, 753), _c1)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 376, 838]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/03_icon_Decrease.png
try:
    _c3 = get_crop(3, 99, 96)
    canvas.paste(_c3, (996, 2444), _c3)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/04_icon_in_schooliKhouRy-coLLEGE.png
try:
    _c4 = get_crop(4, 1440, 312)
    canvas.paste(_c4, (0, 0), _c4)
except Exception:
    pass
layout["in_schooliKhouRy-coLLEGE"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 102)
    canvas.paste(_c6, (1107, 2442), _c6)
except Exception:
    pass
layout["icon_6"] = [1107, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/07_icon_Share.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 60)
    canvas.paste(_c8, (315, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [315, 3, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/09_icon_Science_Technology_._High_Tech.png
try:
    _c9 = get_crop(9, 75, 72)
    canvas.paste(_c9, (249, 2588), _c9)
except Exception:
    pass
layout["Science_&_Technology_._Hi"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 59)
    canvas.paste(_c10, (248, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [248, 3, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/11_icon_9.38.png
try:
    _c11 = get_crop(11, 55, 60)
    canvas.paste(_c11, (181, 2), _c11)
except Exception:
    pass
layout["9.38"] = [181, 2, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/12_icon_9.38.png
try:
    _c12 = get_crop(12, 54, 61)
    canvas.paste(_c12, (115, 2), _c12)
except Exception:
    pass
layout["9.38"] = [115, 2, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 63)
    canvas.paste(_c13, (1318, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1318, 1, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 61, 63)
    canvas.paste(_c14, (1212, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 1, 1273, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 62)
    canvas.paste(_c15, (1262, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1262, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 37, 48)
    canvas.paste(_c16, (196, 594), _c16)
except Exception:
    pass
layout["icon_16"] = [196, 594, 233, 642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 39, 45)
    canvas.paste(_c17, (237, 595), _c17)
except Exception:
    pass
layout["icon_17"] = [237, 595, 276, 640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 61)
    canvas.paste(_c18, (383, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [383, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 32, 46)
    canvas.paste(_c19, (279, 594), _c19)
except Exception:
    pass
layout["icon_19"] = [279, 594, 311, 640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/20_icon_hrs_30_mins.png
try:
    _c20 = get_crop(20, 312, 76)
    canvas.paste(_c20, (118, 1567), _c20)
except Exception:
    pass
layout["hrs_30_mins"] = [118, 1567, 430, 1643]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 38, 45)
    canvas.paste(_c21, (315, 594), _c21)
except Exception:
    pass
layout["icon_21"] = [315, 594, 353, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/22_icon_Northeastern_University.png
try:
    _c22 = get_crop(22, 516, 144)
    canvas.paste(_c22, (144, 1155), _c22)
except Exception:
    pass
layout["Northeastern_University"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/23_icon_The_organizer_will_review_refund_request.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1422), _c23)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/24_text_9.38.png
try:
    _c24 = get_crop(24, 96, 49)
    canvas.paste(_c24, (16, 12), _c24)
except Exception:
    pass
layout["9.38"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/25_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c25 = get_crop(25, 1440, 312)
    canvas.paste(_c25, (0, 0), _c25)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/26_text_Tuesday_March_26.png
try:
    _c26 = get_crop(26, 516, 144)
    canvas.paste(_c26, (144, 1155), _c26)
except Exception:
    pass
layout["Tuesday;_March_26"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/27_text_6.00_PM.png
try:
    _c27 = get_crop(27, 213, 63)
    canvas.paste(_c27, (541, 890), _c27)
except Exception:
    pass
layout["6.00_PM"] = [541, 890, 754, 953]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/28_text_Break_into_Tech_New_York.png
try:
    _c28 = get_crop(28, 516, 144)
    canvas.paste(_c28, (144, 1155), _c28)
except Exception:
    pass
layout["Break_into_Tech:_New_York"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/29_text_About_this_event.png
try:
    _c29 = get_crop(29, 453, 67)
    canvas.paste(_c29, (44, 1982), _c29)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/30_text_Join_our_Break_into_Tech_event_to_discov.png
try:
    _c30 = get_crop(30, 99, 96)
    canvas.paste(_c30, (996, 2444), _c30)
except Exception:
    pass
layout["Join_our_Break_into_Tech_"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/31_text_BiT_Boston.png
try:
    _c31 = get_crop(31, 75, 72)
    canvas.paste(_c31, (249, 2588), _c31)
except Exception:
    pass
layout["BiT_Boston"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/32_text_Free.png
try:
    _c32 = get_crop(32, 105, 48)
    canvas.paste(_c32, (116, 2599), _c32)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_07_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-9/33_clickable_Back.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (36, 108), _c33)
except Exception:
    pass
layout["Back"] = [36, 108, 180, 252]
