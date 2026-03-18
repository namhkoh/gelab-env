# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_17
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19.png
# step_index: 17/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available objects:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw
# - font_sm, font_md, font_lg, font_xl

# 1) Fill overall background (very light off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(249, 250, 251))

# 2) Status bar area at top (~96px)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(212, 212, 212))

# subtle inner highlight to mimic phone status gradient
draw.rectangle([(0, 0), (1440, 6)], fill=(232, 232, 232))

# 3) Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 232
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# header bottom divider
draw.line([(32, header_bottom), (1408, header_bottom)], fill=(235, 235, 238), width=2)

# faint shadow under header
draw.line([(32, header_bottom+2), (1408, header_bottom+2)], fill=(245, 245, 246), width=1)

# 4) Main content separators
# thin divider separating content blocks (approx where testimonials/content ends and card begins)
divider_y = 2360
draw.line([(32, divider_y), (1408, divider_y)], fill=(238, 238, 240), width=2)

# small secondary divider a bit above the card area
draw.line([(32, divider_y + 12), (1408, divider_y + 12)], fill=(250, 250, 250), width=1)

# 5) Blue-outlined ticket/card area with rounded corners (complimentary access card)
card_left = 56
card_right = 1384
card_top = divider_y + 28   # start just below the divider
card_bottom = 2650
card_radius = 22

# subtle shadow under the card (soft grey block offset)
shadow_offset = 10
draw.rounded_rectangle(
    [(card_left + 4, card_top + shadow_offset, card_right + 4, card_bottom + shadow_offset)],
    radius=card_radius + 2,
    fill=(229, 229, 232)
)

# card inner background (white)
draw.rounded_rectangle(
    [(card_left, card_top, card_right, card_bottom)],
    radius=card_radius,
    fill=(255, 255, 255),
    outline=(49, 86, 255),  # blue border color
    width=6
)

# subtle inner separator line inside the card to visually separate title area from controls
inner_sep_y = card_top + 92
draw.line([(card_left + 24, inner_sep_y), (card_right - 24, inner_sep_y)], fill=(245, 245, 247), width=1)

# 6) Light divider above the bottom area where the Reserve button will appear (do not draw the button)
bottom_div_y = 2728
draw.line([(24, bottom_div_y), (1416, bottom_div_y)], fill=(239, 239, 241), width=2)

# 7) Page-wide subtle footer background band (behind reserve area) - very light to separate from content
footer_band_top = bottom_div_y + 8
footer_band_bottom = 2960
draw.rectangle([(0, footer_band_top), (1440, footer_band_bottom)], fill=(252, 250, 249))

# 8) Additional subtle vertical guides/margins (visual structure, not content)
# left and right content gutter indicators (very faint)
draw.line([(48, header_bottom + 12), (48, footer_band_top - 12)], fill=(250, 250, 250), width=1)
draw.line([(1392, header_bottom + 12), (1392, footer_band_top - 12)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/02_icon_Decrease.png
try:
    _c2 = get_crop(2, 99, 96)
    canvas.paste(_c2, (996, 2444), _c2)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 51, 55)
    canvas.paste(_c3, (316, 7), _c3)
except Exception:
    pass
layout["icon_3"] = [316, 7, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/04_icon_9.13.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 94, 104)
    canvas.paste(_c6, (1107, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1107, 2441, 1201, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/07_icon_Reserve_a_spot.png
try:
    _c7 = get_crop(7, 1296, 132)
    canvas.paste(_c7, (72, 2756), _c7)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 55)
    canvas.paste(_c8, (250, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [250, 6, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 45, 61)
    canvas.paste(_c9, (1156, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1156, 3, 1201, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 96, 59)
    canvas.paste(_c10, (1215, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [1215, 3, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 55)
    canvas.paste(_c11, (182, 6), _c11)
except Exception:
    pass
layout["icon_11"] = [182, 6, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 47, 57)
    canvas.paste(_c12, (1325, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1325, 4, 1372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/13_icon_9.13.png
try:
    _c13 = get_crop(13, 52, 58)
    canvas.paste(_c13, (117, 4), _c13)
except Exception:
    pass
layout["9.13"] = [117, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/14_icon_Minorities_Building_..png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (36, 108), _c14)
except Exception:
    pass
layout["Minorities_Building_."] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 57)
    canvas.paste(_c15, (384, 5), _c15)
except Exception:
    pass
layout["icon_15"] = [384, 5, 431, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/16_icon_Free.png
try:
    _c16 = get_crop(16, 139, 118)
    canvas.paste(_c16, (96, 2568), _c16)
except Exception:
    pass
layout["Free"] = [96, 2568, 235, 2686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/17_icon_Minorities_Building_..png
try:
    _c17 = get_crop(17, 97, 62)
    canvas.paste(_c17, (289, 234), _c17)
except Exception:
    pass
layout["Minorities_Building_."] = [289, 234, 386, 296]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/18_icon_Free.png
try:
    _c18 = get_crop(18, 75, 72)
    canvas.paste(_c18, (249, 2588), _c18)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/19_text_9.13.png
try:
    _c19 = get_crop(19, 91, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/20_text_Always_forward-thinking_Andre_is_current.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1116, 108), _c20)
except Exception:
    pass
layout["Always_forward-thinking,_"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/21_text_emerging_technologies_of_the_Metaverse_A.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["emerging_technologies_of_"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/22_text_digital_landscape._His_relentless_pursui.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1116, 108), _c22)
except Exception:
    pass
layout["digital_landscape._His_re"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/23_text_Testimonials_From_Attendees.png
try:
    _c23 = get_crop(23, 634, 57)
    canvas.paste(_c23, (42, 786), _c23)
except Exception:
    pass
layout["Testimonials_From_Attende"] = [42, 786, 676, 843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/24_text_The_presentation_each_of_you_on_the_team.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1116, 108), _c24)
except Exception:
    pass
layout["'The_presentation_each_of"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/25_text_was_delivered_enthusiastically_and_provi.png
try:
    _c25 = get_crop(25, 1254, 74)
    canvas.paste(_c25, (44, 1032), _c25)
except Exception:
    pass
layout["was_delivered_enthusiasti"] = [44, 1032, 1298, 1106]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/26_text_presentation_particularly_on_the_areas_t.png
try:
    _c26 = get_crop(26, 1310, 65)
    canvas.paste(_c26, (40, 1163), _c26)
except Exception:
    pass
layout["presentation,_particularl"] = [40, 1163, 1350, 1228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/27_text_capital_as_well_as_what_to_avoid..png
try:
    _c27 = get_crop(27, 676, 61)
    canvas.paste(_c27, (41, 1229), _c27)
except Exception:
    pass
layout["capital;_as_well_as_what_"] = [41, 1229, 717, 1290]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/28_text_Thank_you_again..png
try:
    _c28 = get_crop(28, 376, 64)
    canvas.paste(_c28, (40, 1353), _c28)
except Exception:
    pass
layout["Thank_you_again.'"] = [40, 1353, 416, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/29_text_Valerie_Best.png
try:
    _c29 = get_crop(29, 260, 52)
    canvas.paste(_c29, (449, 1355), _c29)
except Exception:
    pass
layout["Valerie_Best"] = [449, 1355, 709, 1407]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/30_text_Partners.png
try:
    _c30 = get_crop(30, 209, 61)
    canvas.paste(_c30, (42, 1543), _c30)
except Exception:
    pass
layout["Partners:"] = [42, 1543, 251, 1604]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/31_text_MFV.png
try:
    _c31 = get_crop(31, 103, 45)
    canvas.paste(_c31, (73, 1673), _c31)
except Exception:
    pass
layout["MFV"] = [73, 1673, 176, 1718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/32_text_Franchise_Expo_West.png
try:
    _c32 = get_crop(32, 451, 67)
    canvas.paste(_c32, (71, 1794), _c32)
except Exception:
    pass
layout["Franchise_Expo_West"] = [71, 1794, 522, 1861]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/33_text_ICABA.png
try:
    _c33 = get_crop(33, 144, 52)
    canvas.paste(_c33, (70, 1922), _c33)
except Exception:
    pass
layout["ICABA"] = [70, 1922, 214, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/34_text_Sociallybuzz.png
try:
    _c34 = get_crop(34, 278, 67)
    canvas.paste(_c34, (67, 2045), _c34)
except Exception:
    pass
layout["Sociallybuzz"] = [67, 2045, 345, 2112]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/35_text_REGISTER_TODAY_TO_ALSO_GET_FULL_ACCESS_T.png
try:
    _c35 = get_crop(35, 99, 96)
    canvas.paste(_c35, (996, 2444), _c35)
except Exception:
    pass
layout["REGISTER_TODAY_TO_ALSO_GE"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/36_text_FRANCHISE_EXPO_WEST_EXHIBIT_HALL..png
try:
    _c36 = get_crop(36, 75, 72)
    canvas.paste(_c36, (249, 2588), _c36)
except Exception:
    pass
layout["FRANCHISE_EXPO_WEST_EXHIB"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_17_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-19/37_text_Complimentary_Access.png
try:
    _c37 = get_crop(37, 75, 72)
    canvas.paste(_c37, (249, 2588), _c37)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]
