# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_11
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13.png
# step_index: 11/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Event page
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Overall page background (very light off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FAFAFB")

# Status bar (top area) - subtle grey to separate system info row
status_h = 110
draw.rectangle([(0, 0), (1440, status_h)], fill="#E9E9EB")
draw.line([(0, status_h), (1440, status_h)], fill="#D8D6DB", width=1)

# Header / toolbar background (under status bar)
header_top = status_h
header_bottom = 240
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# faint divider under header
draw.line([(40, header_bottom), (1400, header_bottom)], fill="#E7E3EA", width=1)

# Subtle title banner behind the page title (keeps icons/text legible)
draw.rounded_rectangle([(200, 130), (1240, 210)], radius=20, fill="#F6F4FB")

# Large subtle content band behind the description area (keeps long text blocks readable)
draw.rectangle([(48, 240), (1392, 1200)], fill="#FFFFFF")  # main content stays white

# Separator lines between major content sections
sep_x0, sep_x1 = 48, 1392
# after description block
draw.line([(sep_x0, 1220), (sep_x1, 1220)], fill="#EFEAF0", width=2)
# location section divider (above location)
draw.line([(sep_x0, 1480), (sep_x1, 1480)], fill="#EFEAF0", width=2)
# divider below location / above organizer
draw.line([(sep_x0, 1680), (sep_x1, 1680)], fill="#EFEAF0", width=2)
# divider above tickets area
draw.line([(sep_x0, 2240), (sep_x1, 2240)], fill="#EFEAF0", width=2)

# Organizer section background (soft subtle card-like region)
org_top, org_bottom = 1960, 2160
draw.rounded_rectangle([(48, org_top), (1392, org_bottom)], radius=18, fill="#FBF9FF")

# Thin faint horizontal rule above organizer label area (for emphasis)
draw.line([(200, org_top + 8), (1240, org_top + 8)], fill="#F0EDF3", width=1)

# Ticket selection card (rounded rectangle with blue outline and slight shadow)
card_left, card_top = 48, 2320
card_right, card_bottom = 1392, 2630
# subtle shadow layer (offset)
draw.rounded_rectangle([(card_left + 6, card_top + 8), (card_right + 6, card_bottom + 8)],
                       radius=28, fill="#ECEFF8")
# main card
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=28, fill="#FFFFFF", outline="#2F57E6", width=8)
# faint inner separator inside the card (to visually divide title row from details)
draw.line([(card_left + 24, card_top + 180), (card_right - 24, card_top + 180)], fill="#EEF1F7", width=1)

# Light area above the bottom CTA to provide spacing and a subtle divider/shadow
draw.rectangle([(0, 2688), (1440, 2736)], fill="#FFFFFF")
draw.line([(48, 2688), (1392, 2688)], fill="#ECE7EB", width=1)

# Footer safe area background (keeps space for the reserve CTA that will be pasted)
draw.rectangle([(0, 2736), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/02_icon_9.10.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["9.10"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2444), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 93, 102)
    canvas.paste(_c5, (1108, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2442, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 56)
    canvas.paste(_c6, (316, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [316, 6, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/07_icon_Reserve_a_spot.png
try:
    _c7 = get_crop(7, 1296, 132)
    canvas.paste(_c7, (72, 2756), _c7)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 44, 56)
    canvas.paste(_c8, (1326, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 95, 56)
    canvas.paste(_c9, (1218, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [1218, 4, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 55)
    canvas.paste(_c10, (249, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [249, 6, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 51, 55)
    canvas.paste(_c11, (184, 6), _c11)
except Exception:
    pass
layout["icon_11"] = [184, 6, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/12_icon_Union_Square_Wine_Spirits_140_Ath_Avenue.png
try:
    _c12 = get_crop(12, 541, 144)
    canvas.paste(_c12, (450, 2148), _c12)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [450, 2148, 991, 2292]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/13_icon_9.10.png
try:
    _c13 = get_crop(13, 51, 58)
    canvas.paste(_c13, (118, 4), _c13)
except Exception:
    pass
layout["9.10"] = [118, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/14_icon_Show_map.png
try:
    _c14 = get_crop(14, 226, 144)
    canvas.paste(_c14, (1166, 1501), _c14)
except Exception:
    pass
layout["Show_map"] = [1166, 1501, 1392, 1645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/15_icon_Free.png
try:
    _c15 = get_crop(15, 139, 100)
    canvas.paste(_c15, (98, 2576), _c15)
except Exception:
    pass
layout["Free"] = [98, 2576, 237, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/16_icon_Organized_by.png
try:
    _c16 = get_crop(16, 541, 144)
    canvas.paste(_c16, (450, 2148), _c16)
except Exception:
    pass
layout["Organized_by"] = [450, 2148, 991, 2292]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/17_icon_Union_Square_Wine_Spirits.png
try:
    _c17 = get_crop(17, 206, 144)
    canvas.paste(_c17, (48, 1283), _c17)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/18_icon_Essential_Artist_Series_1_1_launch_event.png
try:
    _c18 = get_crop(18, 206, 144)
    canvas.paste(_c18, (48, 1283), _c18)
except Exception:
    pass
layout["Essential_Artist_Series_1"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/19_icon_Read_less.png
try:
    _c19 = get_crop(19, 206, 144)
    canvas.paste(_c19, (48, 1283), _c19)
except Exception:
    pass
layout["Read_less"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/20_icon_Free.png
try:
    _c20 = get_crop(20, 75, 72)
    canvas.paste(_c20, (249, 2588), _c20)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 58)
    canvas.paste(_c21, (383, 4), _c21)
except Exception:
    pass
layout["icon_21"] = [383, 4, 431, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/22_icon_USQ_Wines_Spirits.png
try:
    _c22 = get_crop(22, 541, 144)
    canvas.paste(_c22, (450, 2148), _c22)
except Exception:
    pass
layout["USQ_Wines_&_Spirits"] = [450, 2148, 991, 2292]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/23_text_9.10.png
try:
    _c23 = get_crop(23, 91, 43)
    canvas.paste(_c23, (20, 17), _c23)
except Exception:
    pass
layout["9.10"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/24_text_Tequila_Artistic_T..png
try:
    _c24 = get_crop(24, 535, 77)
    canvas.paste(_c24, (243, 151), _c24)
except Exception:
    pass
layout["Tequila_&_Artistic_T._"] = [243, 151, 778, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/25_text_boasts_an_impeccable_pedigree_Enjoy_a_co.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1116, 108), _c25)
except Exception:
    pass
layout["boasts_an_impeccable_pedi"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/26_text_to.png
try:
    _c26 = get_crop(26, 52, 40)
    canvas.paste(_c26, (43, 346), _c26)
except Exception:
    pass
layout["to"] = [43, 346, 95, 386]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/27_text_black_peppercorn_followed_by_a_delightfu.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1116, 108), _c27)
except Exception:
    pass
layout["black_peppercorn;_followe"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/28_text_Join_us_for_an_unforgettable_evening_fil.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1116, 108), _c28)
except Exception:
    pass
layout["Join_us_for_an_unforgetta"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/29_text_all_six_in_the_seriesl_savor_the_taste_o.png
try:
    _c29 = get_crop(29, 144, 144)
    canvas.paste(_c29, (1116, 108), _c29)
except Exception:
    pass
layout["all_six_in_the_seriesl)_,"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/30_text_sold_to_Pioneer_Works_Yellin_s_brainchil.png
try:
    _c30 = get_crop(30, 206, 144)
    canvas.paste(_c30, (48, 1283), _c30)
except Exception:
    pass
layout["sold_to_Pioneer_Works,_Ye"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/31_text_cultural_hub.png
try:
    _c31 = get_crop(31, 262, 52)
    canvas.paste(_c31, (42, 904), _c31)
except Exception:
    pass
layout["cultural_hub"] = [42, 904, 304, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/32_text_Location.png
try:
    _c32 = get_crop(32, 246, 63)
    canvas.paste(_c32, (41, 1546), _c32)
except Exception:
    pass
layout["Location"] = [41, 1546, 287, 1609]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_11_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-13/33_text_1800_Tequila_Artistic_Transformation.png
try:
    _c33 = get_crop(33, 75, 72)
    canvas.paste(_c33, (249, 2588), _c33)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [249, 2588, 324, 2660]
