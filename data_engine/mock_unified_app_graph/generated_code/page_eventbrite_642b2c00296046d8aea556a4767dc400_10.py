# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_10
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12.png
# step_index: 10/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant off-white)
bg_color = (250, 251, 252)
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar area (top ~100px) - light gray to host time/signal icons
status_bar_color = (242, 245, 247)
draw.rectangle([(0, 0), (1440, 100)], fill=status_bar_color)

# Header / toolbar area (below status bar)
header_top = 100
header_bottom = 220
header_color = (255, 255, 255)
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_color)

# Header bottom divider
divider_color = (230, 230, 235)
draw.line([(36, header_bottom), (1440-36, header_bottom)], fill=divider_color, width=2)

# Large content separator (above ticket card area)
content_div_y = 2400
draw.line([(36, content_div_y), (1440-36, content_div_y)], fill=divider_color, width=1)

# Ticket selection card background with blue outline (rounded rectangle)
card_x = 72
card_y = 2460
card_w = 1296
card_h = 220
card_radius = 28
card_fill = (255, 255, 255)
card_outline = (60, 87, 255)  # bluish outline
draw.rounded_rectangle(
    [(card_x, card_y), (card_x + card_w, card_y + card_h)],
    radius=card_radius,
    fill=card_fill,
    outline=card_outline,
    width=6
)

# Subtle shadow line under the ticket card
shadow_line_y = card_y + card_h + 8
draw.line([(card_x + 6, shadow_line_y), (card_x + card_w - 6, shadow_line_y)], fill=(220,220,225), width=2)

# Reserve button background (orange rounded rectangle) at bottom
reserve_x = 72
reserve_y = 2756
reserve_w = 1296
reserve_h = 132
reserve_radius = 10
reserve_fill = (199, 64, 23)  # deep orange
draw.rounded_rectangle(
    [(reserve_x, reserve_y), (reserve_x + reserve_w, reserve_y + reserve_h)],
    radius=reserve_radius,
    fill=reserve_fill
)

# Thin top divider above reserve button to separate from ticket card
draw.line([(36, reserve_y - 28), (1440-36, reserve_y - 28)], fill=(235,235,238), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/02_icon_9.10.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["9.10"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/03_icon_Decrease.png
try:
    _c3 = get_crop(3, 99, 96)
    canvas.paste(_c3, (996, 2444), _c3)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 57)
    canvas.paste(_c4, (316, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 6, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 92, 103)
    canvas.paste(_c6, (1108, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 56)
    canvas.paste(_c7, (250, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [250, 5, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 56)
    canvas.paste(_c8, (183, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [183, 5, 234, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 56)
    canvas.paste(_c9, (1326, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [1326, 5, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/10_icon_Reserve_a_spot.png
try:
    _c10 = get_crop(10, 1296, 132)
    canvas.paste(_c10, (72, 2756), _c10)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 85, 57)
    canvas.paste(_c11, (1218, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1218, 3, 1303, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/12_icon_9.10.png
try:
    _c12 = get_crop(12, 51, 60)
    canvas.paste(_c12, (118, 3), _c12)
except Exception:
    pass
layout["9.10"] = [118, 3, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/13_icon_Free.png
try:
    _c13 = get_crop(13, 136, 117)
    canvas.paste(_c13, (97, 2568), _c13)
except Exception:
    pass
layout["Free"] = [97, 2568, 233, 2685]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/14_icon_Tequila_Artistic_T...png
try:
    _c14 = get_crop(14, 542, 84)
    canvas.paste(_c14, (246, 141), _c14)
except Exception:
    pass
layout["Tequila_&_Artistic_T.."] = [246, 141, 788, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 60)
    canvas.paste(_c15, (384, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [384, 3, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/16_icon_Free.png
try:
    _c16 = get_crop(16, 75, 72)
    canvas.paste(_c16, (249, 2588), _c16)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/17_text_9.10.png
try:
    _c17 = get_crop(17, 91, 43)
    canvas.paste(_c17, (20, 17), _c17)
except Exception:
    pass
layout["9.10"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/18_text_Unveiling_the_1800_Essential_Artist_Seri.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1116, 108), _c18)
except Exception:
    pass
layout["Unveiling_the_1800_Essent"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/19_text_Calling_all_art_enthusiasts_tequila_afic.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1116, 108), _c19)
except Exception:
    pass
layout["Calling_all_art_enthusias"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/20_text_host_a.png
try:
    _c20 = get_crop(20, 133, 52)
    canvas.paste(_c20, (44, 574), _c20)
except Exception:
    pass
layout["host_a"] = [44, 574, 177, 626]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/21_text_Tequila_Essential_Artist_Series_11_..png
try:
    _c21 = get_crop(21, 704, 65)
    canvas.paste(_c21, (41, 631), _c21)
except Exception:
    pass
layout["Tequila_Essential_Artist_"] = [41, 631, 745, 696]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/22_text_This_limited-edition_series_features_the.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (36, 108), _c22)
except Exception:
    pass
layout["This_limited-edition_seri"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/23_text_thought-provoking_pieces._Each_bottle_is.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1116, 108), _c23)
except Exception:
    pass
layout["thought-provoking_pieces."] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/24_text_have_a_special_guest_appearance_by_the_a.png
try:
    _c24 = get_crop(24, 1251, 73)
    canvas.paste(_c24, (41, 1007), _c24)
except Exception:
    pass
layout["have_a_special_guest_appe"] = [41, 1007, 1292, 1080]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/25_text_But_the_beauty_goes_beyond_the_label._Cr.png
try:
    _c25 = get_crop(25, 1213, 65)
    canvas.paste(_c25, (43, 1202), _c25)
except Exception:
    pass
layout["But_the_beauty_goes_beyon"] = [43, 1202, 1256, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/26_text_boasts_an_impeccable_pedigree._Enjoy_a_c.png
try:
    _c26 = get_crop(26, 1325, 63)
    canvas.paste(_c26, (42, 1329), _c26)
except Exception:
    pass
layout["boasts_an_impeccable_pedi"] = [42, 1329, 1367, 1392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/27_text_to.png
try:
    _c27 = get_crop(27, 52, 40)
    canvas.paste(_c27, (43, 1403), _c27)
except Exception:
    pass
layout["to"] = [43, 1403, 95, 1443]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/28_text_black_peppercorn_followed_by_a_delightfu.png
try:
    _c28 = get_crop(28, 1252, 66)
    canvas.paste(_c28, (41, 1453), _c28)
except Exception:
    pass
layout["black_peppercorn,_followe"] = [41, 1453, 1293, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/29_text_Join_us_for_an_unforgettable_evening_fil.png
try:
    _c29 = get_crop(29, 99, 96)
    canvas.paste(_c29, (996, 2444), _c29)
except Exception:
    pass
layout["Join_us_for_an_unforgetta"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/30_text_and_of_course.png
try:
    _c30 = get_crop(30, 310, 57)
    canvas.paste(_c30, (42, 1709), _c30)
except Exception:
    pass
layout["and_of_course,"] = [42, 1709, 352, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/31_text_all_six_in_the_seriesl_savor_the_taste_o.png
try:
    _c31 = get_crop(31, 99, 96)
    canvas.paste(_c31, (996, 2444), _c31)
except Exception:
    pass
layout["all_six_in_the_seriesl)_,"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/32_text_sold_to_Pioneer_Works_Yellin_s_brainchil.png
try:
    _c32 = get_crop(32, 99, 96)
    canvas.paste(_c32, (996, 2444), _c32)
except Exception:
    pass
layout["sold_to_Pioneer_Works,_Ye"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/33_text_cultural_hub.png
try:
    _c33 = get_crop(33, 262, 52)
    canvas.paste(_c33, (42, 1961), _c33)
except Exception:
    pass
layout["cultural_hub"] = [42, 1961, 304, 2013]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/34_text_Don_t_miss_this_unique_opportunity_to_ex.png
try:
    _c34 = get_crop(34, 99, 96)
    canvas.paste(_c34, (996, 2444), _c34)
except Exception:
    pass
layout["Don't_miss_this_unique_op"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/35_text_Essential_Artist_Series_1_1_launch_event.png
try:
    _c35 = get_crop(35, 99, 96)
    canvas.paste(_c35, (996, 2444), _c35)
except Exception:
    pass
layout["Essential_Artist_Series_1"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_10_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-12/36_text_1800_Tequila_Artistic_Transformation.png
try:
    _c36 = get_crop(36, 75, 72)
    canvas.paste(_c36, (249, 2588), _c36)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [249, 2588, 324, 2660]
