# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_09
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11.png
# step_index: 9/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the canvas (1440x2960)
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# 1) Base background (dominant color - white)
draw.rectangle([0, 0, 1440, 2960], fill=(255, 255, 255))

# 2) Status bar area at top (~84px) - light gray background to match screenshot status bar
status_h = 84
draw.rectangle([0, 0, 1440, status_h], fill=(235, 235, 235))

# subtle bottom divider under status bar
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill=(210, 210, 210), width=1)

# 3) Header / toolbar area beneath status bar (approx 84px -> 200px)
header_top = status_h
header_bottom = 200
# keep header visually white but add a very subtle shadow line to separate from content
draw.rectangle([0, header_top, 1440, header_bottom], fill=(255, 255, 255))
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(235, 235, 235), width=1)

# 4) Main content separators and subtle section divider lines
# Divider under the top info block (around refund policy area in screenshot)
draw.line([(40, 620), (1400, 620)], fill=(230, 230, 230), width=2)

# Another thin separator further down to visually break content areas
draw.line([(40, 1280), (1400, 1280)], fill=(245, 245, 245), width=1)

# 5) "About this event" header area left intact (no text drawing) - add an accent horizontal rule below its title
about_title_y = 842  # approximate y for the "About this event" section heading
draw.line([(40, about_title_y + 72), (1400, about_title_y + 72)], fill=(240, 240, 240), width=1)

# 6) Ticket selection card (rounded rectangle) near bottom of page
# Draw a subtle shadow, then the white card with a colored outline (blue) to reflect the UI structure
card_x1 = 48
card_x2 = 1392
card_top = 2360
card_bottom = 2576
card_radius = 28

# shadow (slightly offset, light gray)
shadow_offset = 8
draw.rounded_rectangle(
    [card_x1 + shadow_offset, card_top + shadow_offset, card_x2 + shadow_offset, card_bottom + shadow_offset],
    radius=card_radius,
    fill=(240, 240, 240),
    outline=None
)

# card background (white)
draw.rounded_rectangle(
    [card_x1, card_top, card_x2, card_bottom],
    radius=card_radius,
    fill=(255, 255, 255),
    outline=None
)

# card border (blue outline)
outline_color = (57, 86, 255)  # bright blue-ish border
draw.rounded_rectangle(
    [card_x1 + 4, card_top + 4, card_x2 - 4, card_bottom - 4],
    radius=card_radius,
    outline=outline_color,
    width=8
)

# 7) Thin divider above the ticket card to separate content from card
draw.line([(40, card_top - 24), (1400, card_top - 24)], fill=(235, 235, 235), width=1)

# 8) Reserve button area: leave exact button artwork to be pasted, but provide a subtle top spacing rule
reserve_button_top = 2728
draw.line([(40, reserve_button_top - 20), (1400, reserve_button_top - 20)], fill=(250, 250, 250), width=1)

# 9) Bottom safe area / footer background (very light) to ground the "Reserve a spot" area
draw.rectangle([0, 2728, 1440, 2960], fill=(255, 255, 255))

# 10) Final subtle left and right content gutters (visual guides, very faint)
draw.line([(40, header_bottom + 12), (40, 2600)], fill=(250, 250, 250), width=1)
draw.line([(1400, header_bottom + 12), (1400, 2600)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/02_icon_Decrease.png
try:
    _c2 = get_crop(2, 99, 96)
    canvas.paste(_c2, (996, 2444), _c2)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 92, 104)
    canvas.paste(_c4, (1108, 2441), _c4)
except Exception:
    pass
layout["icon_4"] = [1108, 2441, 1200, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 56)
    canvas.paste(_c5, (316, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [316, 6, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1296, 132)
    canvas.paste(_c6, (72, 2756), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/07_icon_2_hrs.png
try:
    _c7 = get_crop(7, 202, 77)
    canvas.paste(_c7, (46, 425), _c7)
except Exception:
    pass
layout["2_hrs"] = [46, 425, 248, 502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/08_icon_9.10.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["9.10"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 57)
    canvas.paste(_c9, (248, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [248, 4, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/10_icon_Food_Drink_._Spirits.png
try:
    _c10 = get_crop(10, 474, 99)
    canvas.paste(_c10, (40, 930), _c10)
except Exception:
    pass
layout["Food_&_Drink_._Spirits"] = [40, 930, 514, 1029]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/11_icon_9.10.png
try:
    _c11 = get_crop(11, 54, 58)
    canvas.paste(_c11, (182, 3), _c11)
except Exception:
    pass
layout["9.10"] = [182, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 56)
    canvas.paste(_c12, (1325, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [1325, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 96, 58)
    canvas.paste(_c13, (1217, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1217, 2, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/14_icon_9.10.png
try:
    _c14 = get_crop(14, 51, 59)
    canvas.paste(_c14, (118, 3), _c14)
except Exception:
    pass
layout["9.10"] = [118, 3, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/15_icon_Free.png
try:
    _c15 = get_crop(15, 75, 72)
    canvas.paste(_c15, (249, 2588), _c15)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/16_icon_Free.png
try:
    _c16 = get_crop(16, 146, 116)
    canvas.paste(_c16, (91, 2567), _c16)
except Exception:
    pass
layout["Free"] = [91, 2567, 237, 2683]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/17_icon_Refund_policy.png
try:
    _c17 = get_crop(17, 298, 73)
    canvas.paste(_c17, (133, 535), _c17)
except Exception:
    pass
layout["Refund_policy"] = [133, 535, 431, 608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/18_icon_Union_Square_Wine_Spirits.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 281), _c18)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/19_text_9.10.png
try:
    _c19 = get_crop(19, 91, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.10"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/20_text_Tequila_Artistic_T.png
try:
    _c20 = get_crop(20, 1344, 144)
    canvas.paste(_c20, (48, 281), _c20)
except Exception:
    pass
layout["Tequila_&_Artistic_T_"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 281), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/22_text_About_this_event.png
try:
    _c22 = get_crop(22, 452, 63)
    canvas.paste(_c22, (45, 842), _c22)
except Exception:
    pass
layout["About_this_event"] = [45, 842, 497, 905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/23_text_Unveiling_the_1800_Essential_Artist_Seri.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 281), _c23)
except Exception:
    pass
layout["Unveiling_the_1800_Essent"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/24_text_Calling_all_art_enthusiasts_tequila_afic.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 281), _c24)
except Exception:
    pass
layout["Calling_all_art_enthusias"] = [48, 281, 1392, 425]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/25_text_host_a.png
try:
    _c25 = get_crop(25, 133, 52)
    canvas.paste(_c25, (44, 1334), _c25)
except Exception:
    pass
layout["host_a"] = [44, 1334, 177, 1386]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/26_text_Tequila_Essential_Artist_Series_11_..png
try:
    _c26 = get_crop(26, 702, 63)
    canvas.paste(_c26, (43, 1394), _c26)
except Exception:
    pass
layout["Tequila_Essential_Artist_"] = [43, 1394, 745, 1457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/27_text_This_limited-edition_series_features_the.png
try:
    _c27 = get_crop(27, 1064, 63)
    canvas.paste(_c27, (41, 1523), _c27)
except Exception:
    pass
layout["This_limited-edition_seri"] = [41, 1523, 1105, 1586]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/28_text_thought-provoking_pieces._Each_bottle_is.png
try:
    _c28 = get_crop(28, 99, 96)
    canvas.paste(_c28, (996, 2444), _c28)
except Exception:
    pass
layout["thought-provoking_pieces."] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/29_text_have_a_special_guest_appearance_by_the_a.png
try:
    _c29 = get_crop(29, 99, 96)
    canvas.paste(_c29, (996, 2444), _c29)
except Exception:
    pass
layout["have_a_special_guest_appe"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/30_text_But_the_beauty_goes_beyond_the_label._Cr.png
try:
    _c30 = get_crop(30, 99, 96)
    canvas.paste(_c30, (996, 2444), _c30)
except Exception:
    pass
layout["But_the_beauty_goes_beyon"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/31_text_boasts_an_impeccable_pedigree._Enjoy_a_c.png
try:
    _c31 = get_crop(31, 99, 96)
    canvas.paste(_c31, (996, 2444), _c31)
except Exception:
    pass
layout["boasts_an_impeccable_pedi"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/32_text_to.png
try:
    _c32 = get_crop(32, 52, 41)
    canvas.paste(_c32, (43, 2163), _c32)
except Exception:
    pass
layout["to"] = [43, 2163, 95, 2204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/33_text_black_peppercorn_followed_by_a_delightfu.png
try:
    _c33 = get_crop(33, 99, 96)
    canvas.paste(_c33, (996, 2444), _c33)
except Exception:
    pass
layout["black_peppercorn;_followe"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_09_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-11/34_text_1800_Tequila_Artistic_Transformation.png
try:
    _c34 = get_crop(34, 75, 72)
    canvas.paste(_c34, (249, 2588), _c34)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [249, 2588, 324, 2660]
