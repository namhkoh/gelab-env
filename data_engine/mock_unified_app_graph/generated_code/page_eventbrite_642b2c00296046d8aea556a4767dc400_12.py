# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_12
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14.png
# step_index: 12/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (light off-white / very pale gray-blue)
draw.rectangle((0, 0, 1440, 2960), fill=(247, 250, 250))

# Status bar (top area) - dark muted color
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(60, 66, 72))

# Pale green notification/toast banner below status bar
toast_y0 = status_h + 8
toast_y1 = toast_y0 + 84
draw.rectangle((0, toast_y0, 1440, toast_y1), fill=(230, 247, 239))
# subtle bottom border for the toast
draw.line((0, toast_y1, 1440, toast_y1), fill=(208, 234, 218), width=1)

# Header area (subtle white band) with bottom divider
header_y0 = toast_y1
header_y1 = header_y0 + 80
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))
draw.line((32, header_y1, 1440-32, header_y1), fill=(235, 235, 235), width=1)

# Main content separators (approximate positions matching screenshot structure)
# divider below long text area / "Read less"
draw.line((40, 1400, 1440-40, 1400), fill=(240, 240, 242), width=2)
# divider under Location section
draw.line((40, 1620, 1440-40, 1620), fill=(240, 240, 242), width=1)
# divider above Organized by area
draw.line((40, 1960, 1440-40, 1960), fill=(245, 245, 247), width=1)
# faint divider above ticket card area
ticket_top = 2320
draw.line((40, ticket_top-16, 1440-40, ticket_top-16), fill=(238, 238, 240), width=1)

# Organizer area background hint (center band with slightly different tint)
org_y0 = 2000
org_y1 = 2200
draw.rectangle((0, org_y0, 1440, org_y1), fill=(255, 255, 255))
# small top/bottom separators for organizer area
draw.line((48, org_y0+12, 1440-48, org_y0+12), fill=(245,245,247), width=1)
draw.line((48, org_y1-12, 1440-48, org_y1-12), fill=(245,245,247), width=1)

# Ticket selection card (rounded rectangle with colored outline)
card_x0 = 48
card_x1 = 1440 - 48
card_y0 = ticket_top
card_y1 = 2640
card_radius = 28

# shadow (soft simulated by a slightly offset, very light gray rounded rect)
shadow_offset = 8
draw.rounded_rectangle(
    (card_x0 + shadow_offset, card_y0 + shadow_offset, card_x1 + shadow_offset, card_y1 + shadow_offset),
    radius=card_radius + 2,
    fill=(243, 244, 246)
)

# main card fill
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=card_radius, fill=(255, 255, 255))

# card border (blue/purple tone)
border_width = 6
# Draw outer border by drawing slightly larger rounded rectangle stroke simulation
for i in range(border_width):
    draw.rounded_rectangle(
        (card_x0 - i, card_y0 - i, card_x1 + i, card_y1 + i),
        radius=card_radius + i,
        outline=(43, 71, 214)
    )

# Inner subtle divider line within the card to hint separation between title area and details
inner_div_y = card_y0 + 110
draw.line((card_x0 + 28, inner_div_y, card_x1 - 28, inner_div_y), fill=(242, 243, 247), width=1)

# Small subtle horizontal guideline above the reserve area (so pasted button sits on a clean band)
draw.line((40, card_y1 + 24, 1440-40, card_y1 + 24), fill=(240,240,242), width=1)

# Additional faint separators through the page to match screenshot structure
separator_positions = [880, 1160, 1960, 2240]
for y in separator_positions:
    draw.line((40, y, 1440-40, y), fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/00_icon_Increase.png
try:
    _c0 = get_crop(0, 96, 96)
    canvas.paste(_c0, (1224, 2444), _c0)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 93, 102)
    canvas.paste(_c2, (1108, 2442), _c2)
except Exception:
    pass
layout["icon_2"] = [1108, 2442, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1296, 132)
    canvas.paste(_c3, (72, 2756), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 57)
    canvas.paste(_c4, (316, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 6, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 44, 57)
    canvas.paste(_c5, (1326, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [1326, 5, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/06_icon_9.10.png
try:
    _c6 = get_crop(6, 50, 60)
    canvas.paste(_c6, (118, 3), _c6)
except Exception:
    pass
layout["9.10"] = [118, 3, 168, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 57)
    canvas.paste(_c7, (185, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [185, 4, 234, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 56)
    canvas.paste(_c8, (248, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [248, 5, 300, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/09_icon_Share.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 108), _c9)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/10_icon_Show_map.png
try:
    _c10 = get_crop(10, 226, 144)
    canvas.paste(_c10, (1166, 1501), _c10)
except Exception:
    pass
layout["Show_map"] = [1166, 1501, 1392, 1645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/11_icon_Free.png
try:
    _c11 = get_crop(11, 141, 100)
    canvas.paste(_c11, (96, 2576), _c11)
except Exception:
    pass
layout["Free"] = [96, 2576, 237, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 42, 55)
    canvas.paste(_c12, (1272, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [1272, 6, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 63, 60)
    canvas.paste(_c13, (1214, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1214, 3, 1277, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/14_icon_Organized_by.png
try:
    _c14 = get_crop(14, 541, 144)
    canvas.paste(_c14, (450, 2148), _c14)
except Exception:
    pass
layout["Organized_by"] = [450, 2148, 991, 2292]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/15_icon_Union_Square_Wine_Spirits_140_Ath_Avenue.png
try:
    _c15 = get_crop(15, 541, 144)
    canvas.paste(_c15, (450, 2148), _c15)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [450, 2148, 991, 2292]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/16_icon_Read_less.png
try:
    _c16 = get_crop(16, 206, 144)
    canvas.paste(_c16, (48, 1283), _c16)
except Exception:
    pass
layout["Read_less"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/17_icon_Free.png
try:
    _c17 = get_crop(17, 75, 72)
    canvas.paste(_c17, (249, 2588), _c17)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/18_icon_Essential_Artist_Series_1_1_launch_event.png
try:
    _c18 = get_crop(18, 206, 144)
    canvas.paste(_c18, (48, 1283), _c18)
except Exception:
    pass
layout["Essential_Artist_Series_1"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/19_icon_Union_Square_Wine_Spirits.png
try:
    _c19 = get_crop(19, 206, 144)
    canvas.paste(_c19, (48, 1283), _c19)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 62)
    canvas.paste(_c20, (383, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [383, 1, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/21_text_9.10.png
try:
    _c21 = get_crop(21, 91, 43)
    canvas.paste(_c21, (20, 17), _c21)
except Exception:
    pass
layout["9.10"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/22_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c22 = get_crop(22, 1440, 312)
    canvas.paste(_c22, (0, 0), _c22)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/23_text_C.png
try:
    _c23 = get_crop(23, 50, 19)
    canvas.paste(_c23, (519, 308), _c23)
except Exception:
    pass
layout["C~"] = [519, 308, 569, 327]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/24_text_to.png
try:
    _c24 = get_crop(24, 52, 40)
    canvas.paste(_c24, (43, 346), _c24)
except Exception:
    pass
layout["to"] = [43, 346, 95, 386]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/25_text_black_peppercorn_followed_by_a_delightfu.png
try:
    _c25 = get_crop(25, 1440, 312)
    canvas.paste(_c25, (0, 0), _c25)
except Exception:
    pass
layout["black_peppercorn;_followe"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/26_text_Join_us_for_an_unforgettable_evening_fil.png
try:
    _c26 = get_crop(26, 1440, 312)
    canvas.paste(_c26, (0, 0), _c26)
except Exception:
    pass
layout["Join_us_for_an_unforgetta"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/27_text_all_six_in_the_seriesl_savor_the_taste_o.png
try:
    _c27 = get_crop(27, 1440, 312)
    canvas.paste(_c27, (0, 0), _c27)
except Exception:
    pass
layout["all_six_in_the_seriesl)_,"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/28_text_sold_to_Pioneer_Works_Yellin_s_brainchil.png
try:
    _c28 = get_crop(28, 206, 144)
    canvas.paste(_c28, (48, 1283), _c28)
except Exception:
    pass
layout["sold_to_Pioneer_Works,_Ye"] = [48, 1283, 254, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/29_text_cultural_hub.png
try:
    _c29 = get_crop(29, 262, 52)
    canvas.paste(_c29, (42, 904), _c29)
except Exception:
    pass
layout["cultural_hub"] = [42, 904, 304, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/30_text_Location.png
try:
    _c30 = get_crop(30, 246, 63)
    canvas.paste(_c30, (41, 1546), _c30)
except Exception:
    pass
layout["Location"] = [41, 1546, 287, 1609]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/31_text_1800_Tequila_Artistic_Transformation.png
try:
    _c31 = get_crop(31, 75, 72)
    canvas.paste(_c31, (249, 2588), _c31)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_12_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-14/32_clickable_Back.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (36, 108), _c32)
except Exception:
    pass
layout["Back"] = [36, 108, 180, 252]
