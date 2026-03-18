# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_06
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8.png
# step_index: 6/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Event page

# Top status bar
status_h = 90
draw.rectangle([(0, 0), (1440, status_h)], fill="#dcdcdc")  # subtle grey status bar
# bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#cfcfcf", width=1)

# Hero/banner area (subtle horizontal gradient feel)
hero_top = status_h
hero_bottom = 540
# simple vertical gradient blend between two light tones
for i, y in enumerate(range(hero_top, hero_bottom)):
    t = i / max(1, (hero_bottom - hero_top - 1))
    # interpolate between two colors (#f8f9fb -> #efeff4)
    r = int(248 * (1 - t) + 239 * t)
    g = int(249 * (1 - t) + 239 * t)
    b = int(251 * (1 - t) + 244 * t)
    draw.line([(0, y), (1440, y)], fill=(r, g, b))

# soft shadow at bottom of hero area
draw.rectangle([(0, hero_bottom - 6), (1440, hero_bottom)], fill="#e6e6e9")

# Main content area background (keeps canvas mostly white but add subtle warm background band)
content_top = hero_bottom
draw.rectangle([(0, content_top), (1440, 2960)], fill="#ffffff")

# Large organizer/follow card (rounded light panel behind organizer info & Follow button)
card_x0 = 48
card_x1 = 1392
card_y0 = 1120
card_y1 = 1288
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=28, fill="#f6f6fb", outline="#e9e8f3", width=2)

# Thin separator between info sections
sep_y = 1680
draw.line([(48, sep_y), (1392, sep_y)], fill="#efeff2", width=2)

# Another faint divider a bit lower
sep_y2 = 2030
draw.line([(48, sep_y2), (1392, sep_y2)], fill="#f2f2f4", width=1)

# Ticket selection card (white card with prominent blue border)
ticket_x0 = 40
ticket_x1 = 1400
ticket_y0 = 1960
ticket_y1 = 2240
draw.rounded_rectangle([(ticket_x0, ticket_y0), (ticket_x1, ticket_y1)],
                       radius=22, fill="#ffffff", outline="#3558f0", width=8)

# Inner subtle shadow for ticket card (top highlight)
draw.rectangle([(ticket_x0 + 8, ticket_y0 + 8), (ticket_x1 - 8, ticket_y0 + 12)], fill="#f8f9ff")

# Light background band above the ticket card to visually separate sections
band_y0 = 1860
band_y1 = ticket_y0
draw.rectangle([(0, band_y0), (1440, band_y1)], fill="#fbfbfc")

# A subtle bottom shadow at the top of the reserve area (we do not draw the reserve button itself)
reserve_top = 2324
draw.rectangle([(0, reserve_top - 6), (1440, reserve_top)], fill="#efe7e2")

# Overall subtle page left/right margins indicated by edge guide lines (very faint)
draw.line([(24, status_h + 8), (24, 2800)], fill="#ffffff00")  # invisible guide (kept harmless)
draw.line([(1416, status_h + 8), (1416, 2800)], fill="#ffffff00")

# End of structural drawing. (No text/icons drawn — those will be pasted on top.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/01_icon_Food_Drink_._Spirits.png
try:
    _c1 = get_crop(1, 472, 100)
    canvas.paste(_c1, (41, 2071), _c1)
except Exception:
    pass
layout["Food_&_Drink_._Spirits"] = [41, 2071, 513, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 111, 104)
    canvas.paste(_c2, (987, 2440), _c2)
except Exception:
    pass
layout["icon_2"] = [987, 2440, 1098, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/03_icon_9.09.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.09"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1440, 636)
    canvas.paste(_c4, (0, 2324), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 109, 103)
    canvas.paste(_c5, (1215, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1215, 2442, 1324, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/06_icon_More.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 91, 102)
    canvas.paste(_c7, (1109, 2442), _c7)
except Exception:
    pass
layout["icon_7"] = [1109, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/08_icon_Share.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1260, 108), _c8)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/09_icon_Ticket_sales_end_soon.png
try:
    _c9 = get_crop(9, 548, 85)
    canvas.paste(_c9, (40, 752), _c9)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 752, 588, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 98, 63)
    canvas.paste(_c10, (1216, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 56, 64)
    canvas.paste(_c11, (1317, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1317, 1, 1373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/12_icon_Free.png
try:
    _c12 = get_crop(12, 138, 106)
    canvas.paste(_c12, (97, 2573), _c12)
except Exception:
    pass
layout["Free"] = [97, 2573, 235, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/13_icon_Free.png
try:
    _c13 = get_crop(13, 99, 105)
    canvas.paste(_c13, (235, 2576), _c13)
except Exception:
    pass
layout["Free"] = [235, 2576, 334, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/14_icon_Unveiling_the_1800_Essential_Artist_Seri.png
try:
    _c14 = get_crop(14, 1440, 636)
    canvas.paste(_c14, (0, 2324), _c14)
except Exception:
    pass
layout["Unveiling_the_1800_Essent"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 57)
    canvas.paste(_c15, (316, 6), _c15)
except Exception:
    pass
layout["icon_15"] = [316, 6, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/16_icon_5_00_PM.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1116, 108), _c16)
except Exception:
    pass
layout["5:00_PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/17_text_9.09.png
try:
    _c17 = get_crop(17, 91, 43)
    canvas.paste(_c17, (20, 17), _c17)
except Exception:
    pass
layout["9.09"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/18_text_Thursday_March_21.png
try:
    _c18 = get_crop(18, 426, 144)
    canvas.paste(_c18, (144, 1155), _c18)
except Exception:
    pass
layout["Thursday;_March_21"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/19_text_5_00_PM.png
try:
    _c19 = get_crop(19, 209, 56)
    canvas.paste(_c19, (567, 893), _c19)
except Exception:
    pass
layout["5:00_PM"] = [567, 893, 776, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/20_text_Tequila_Artistic_Transformation.png
try:
    _c20 = get_crop(20, 426, 144)
    canvas.paste(_c20, (144, 1155), _c20)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/21_text_USQ_Wines_Spirits.png
try:
    _c21 = get_crop(21, 426, 144)
    canvas.paste(_c21, (144, 1155), _c21)
except Exception:
    pass
layout["USQ_Wines_&_Spirits"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/22_text_24_Followers.png
try:
    _c22 = get_crop(22, 426, 144)
    canvas.paste(_c22, (144, 1155), _c22)
except Exception:
    pass
layout["24_Followers"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/23_text_Union_Square_Wine_Spirits.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1422), _c23)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/24_text_2_hrs.png
try:
    _c24 = get_crop(24, 112, 49)
    canvas.paste(_c24, (141, 1580), _c24)
except Exception:
    pass
layout["2_hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1685), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1422), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_06_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-8/27_text_1800_Tequila_Artistic_Transformation.png
try:
    _c27 = get_crop(27, 1440, 636)
    canvas.paste(_c27, (0, 2324), _c27)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [0, 2324, 1440, 2960]
