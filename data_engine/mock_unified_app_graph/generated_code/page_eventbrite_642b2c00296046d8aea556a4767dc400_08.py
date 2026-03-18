# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_08
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10.png
# step_index: 8/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([0, 0, 1440, 2960], fill=(250, 250, 250))  # subtle off-white page background

# Status bar (top area with time/signal background)
status_h = 88
draw.rectangle([0, 0, 1440, status_h], fill=(236, 236, 236))  # light grey status bar
# subtle bottom hairline under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(220, 220, 220), width=1)

# Header / toolbar area (title area)
header_top = status_h
header_bottom = 220
draw.rectangle([0, header_top, 1440, header_bottom], fill=(250, 250, 250))
# slight shadow / divider under header
draw.line([(36, header_bottom), (1404, header_bottom)], fill=(230, 230, 230), width=1)

# Main content separators (thin dividers between logical sections)
separators = [620, 880, 1280, 1680, 2240]
for y in separators:
    draw.line([(48, y), (1392, y)], fill=(238, 238, 240), width=2)

# "About this event" section area suggestion (no text)
about_top = 680
about_bottom = 960
# give a very subtle background band to separate content blocks
draw.rectangle([48, about_top, 1392, about_bottom], fill=(250, 250, 250))
# inner divider under about block
draw.line([(48, about_bottom), (1392, about_bottom)], fill=(242, 242, 244), width=1)

# Location section background band (subtle)
location_top = 1240
location_bottom = 1480
draw.rectangle([36, location_top, 1404, location_bottom], fill=(250, 250, 250))
# right-side subtle map placeholder area (only background, do not draw the "Show map" button)
map_placeholder_x1 = 1166
map_placeholder_x2 = 1404
map_placeholder_y1 = location_top + 8
map_placeholder_y2 = location_top + 140
draw.rectangle([map_placeholder_x1, map_placeholder_y1, map_placeholder_x2, map_placeholder_y2],
               fill=(249, 249, 250), outline=None)

# Organizer block separator + light background
organizer_top = 1560
organizer_bottom = 2120
draw.rectangle([48, organizer_top, 1392, organizer_bottom], fill=(250, 250, 250))
# subtle rounded area to emphasize organizer card center
org_card_w = 820
org_card_h = 280
org_card_x1 = (1440 - org_card_w) // 2
org_card_x2 = org_card_x1 + org_card_w
org_card_y1 = 1760
org_card_y2 = org_card_y1 + org_card_h
draw.rounded_rectangle([org_card_x1, org_card_y1, org_card_x2, org_card_y2],
                       radius=10, outline=(240, 240, 242), width=1, fill=(255, 255, 255))

# Ticket selection card (rounded rectangle with blue outline, but do NOT draw buttons/icons/labels inside)
ticket_card_x1 = 48
ticket_card_x2 = 1392
ticket_card_y1 = 2360
ticket_card_y2 = 2608
card_radius = 18
# main card background
draw.rounded_rectangle([ticket_card_x1, ticket_card_y1, ticket_card_x2, ticket_card_y2],
                       radius=card_radius, fill=(255, 255, 255),
                       outline=(45, 85, 255), width=6)
# inner subtle divider inside ticket card to separate title and price area
inner_div_y = ticket_card_y1 + 84
draw.line([(ticket_card_x1 + 28, inner_div_y), (ticket_card_x2 - 28, inner_div_y)],
          fill=(244, 244, 246), width=2)

# Light shadow under ticket card (simulated with a soft grey band)
shadow_y1 = ticket_card_y2 + 6
shadow_y2 = shadow_y1 + 8
draw.rectangle([ticket_card_x1 + 8, shadow_y1, ticket_card_x2 - 8, shadow_y2], fill=(245, 245, 247))

# Bottom safe area background (behind the reserve button, do not draw the button itself)
bottom_safe_top = 2720
draw.rectangle([0, bottom_safe_top, 1440, 2960], fill=(250, 250, 250))
# thin top divider above reserved area
draw.line([(48, bottom_safe_top), (1392, bottom_safe_top)], fill=(235, 235, 237), width=1)

# final subtle vertical margin guides (not visible UI, lightly colored to aid paste alignment)
# use very faint lines to not interfere with pasted icons/text
draw.line([(48, 0), (48, 2960)], fill=(255, 255, 255), width=2)
draw.line([(1392, 0), (1392, 2960)], fill=(255, 255, 255), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/03_icon_Decrease.png
try:
    _c3 = get_crop(3, 99, 96)
    canvas.paste(_c3, (996, 2444), _c3)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/04_icon_Food_Drink_._Spirits.png
try:
    _c4 = get_crop(4, 234, 144)
    canvas.paste(_c4, (48, 1136), _c4)
except Exception:
    pass
layout["Food_&_Drink_._Spirits"] = [48, 1136, 282, 1280]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 94, 103)
    canvas.paste(_c6, (1107, 2442), _c6)
except Exception:
    pass
layout["icon_6"] = [1107, 2442, 1201, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/07_icon_Free.png
try:
    _c7 = get_crop(7, 140, 102)
    canvas.paste(_c7, (96, 2574), _c7)
except Exception:
    pass
layout["Free"] = [96, 2574, 236, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/08_icon_9.10.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["9.10"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/09_icon_Free.png
try:
    _c9 = get_crop(9, 75, 72)
    canvas.paste(_c9, (249, 2588), _c9)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 97, 59)
    canvas.paste(_c10, (1216, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 1, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/11_icon_9.10.png
try:
    _c11 = get_crop(11, 54, 58)
    canvas.paste(_c11, (182, 3), _c11)
except Exception:
    pass
layout["9.10"] = [182, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 46, 57)
    canvas.paste(_c12, (1325, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1325, 4, 1371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 59)
    canvas.paste(_c13, (247, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [247, 3, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 51, 58)
    canvas.paste(_c14, (316, 5), _c14)
except Exception:
    pass
layout["icon_14"] = [316, 5, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/15_icon_Show_map.png
try:
    _c15 = get_crop(15, 226, 144)
    canvas.paste(_c15, (1166, 1354), _c15)
except Exception:
    pass
layout["Show_map"] = [1166, 1354, 1392, 1498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/16_icon_Union_Square_Wine_Spirits_140_4th_Avenue.png
try:
    _c16 = get_crop(16, 541, 144)
    canvas.paste(_c16, (450, 2001), _c16)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [450, 2001, 991, 2145]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/17_icon_Unveiling_the_1800_Essential_Artist_Seri.png
try:
    _c17 = get_crop(17, 234, 144)
    canvas.paste(_c17, (48, 1136), _c17)
except Exception:
    pass
layout["Unveiling_the_1800_Essent"] = [48, 1136, 282, 1280]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/18_icon_2_hrs.png
try:
    _c18 = get_crop(18, 208, 74)
    canvas.paste(_c18, (48, 427), _c18)
except Exception:
    pass
layout["2_hrs"] = [48, 427, 256, 501]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/19_text_9.10.png
try:
    _c19 = get_crop(19, 91, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.10"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/20_text_Tequila_Artistic_T.png
try:
    _c20 = get_crop(20, 1344, 144)
    canvas.paste(_c20, (48, 281), _c20)
except Exception:
    pass
layout["Tequila_&_Artistic_T_"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/21_text_Union_Square_Wine_Spirits.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 281), _c21)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/22_text_Refund_policy.png
try:
    _c22 = get_crop(22, 299, 63)
    canvas.paste(_c22, (138, 543), _c22)
except Exception:
    pass
layout["Refund_policy"] = [138, 543, 437, 606]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/23_text_The_organizer_will_review_refund_request.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 281), _c23)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/24_text_Location.png
try:
    _c24 = get_crop(24, 243, 63)
    canvas.paste(_c24, (41, 1398), _c24)
except Exception:
    pass
layout["Location"] = [41, 1398, 284, 1461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/25_text_Organized_by.png
try:
    _c25 = get_crop(25, 541, 144)
    canvas.paste(_c25, (450, 2001), _c25)
except Exception:
    pass
layout["Organized_by"] = [450, 2001, 991, 2145]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/26_text_USQ_Wines_Spirits.png
try:
    _c26 = get_crop(26, 541, 144)
    canvas.paste(_c26, (450, 2001), _c26)
except Exception:
    pass
layout["USQ_Wines_&_Spirits"] = [450, 2001, 991, 2145]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/27_text_25.png
try:
    _c27 = get_crop(27, 78, 55)
    canvas.paste(_c27, (682, 2199), _c27)
except Exception:
    pass
layout["25"] = [682, 2199, 760, 2254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/28_text_Followers.png
try:
    _c28 = get_crop(28, 182, 43)
    canvas.paste(_c28, (630, 2272), _c28)
except Exception:
    pass
layout["Followers"] = [630, 2272, 812, 2315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_08_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-10/29_text_1800_Tequila_Artistic_Transformation.png
try:
    _c29 = get_crop(29, 75, 72)
    canvas.paste(_c29, (249, 2588), _c29)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [249, 2588, 324, 2660]
