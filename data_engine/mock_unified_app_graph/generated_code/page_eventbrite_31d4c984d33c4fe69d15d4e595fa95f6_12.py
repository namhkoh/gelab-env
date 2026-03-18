# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_12
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14.png
# step_index: 12/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill=(247, 249, 252))  # very light bluish-gray background

# Status bar (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill=(195, 195, 195))  # subtle gray status bar
# thin divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(220, 220, 220), width=1)

# Header / toolbar area (search + title area) - leave content area blank for pasted elements
header_y0 = status_h
header_y1 = 220
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))  # white header background
# subtle bottom divider for header
draw.line((48, header_y1, 1392, header_y1), fill=(230, 233, 236), width=2)

# Filter / chips row background area (stay subtle, so icons/text pasted remain prominent)
filters_y0 = 360
filters_y1 = 520
# slightly lighter band behind chips so pasted chip pills will sit on it (not drawing the chips themselves)
draw.rectangle((0, filters_y0, 1440, filters_y1), fill=(247, 249, 252))
draw.line((48, filters_y1, 1392, filters_y1), fill=(235, 238, 241), width=1)

# Separator line under filters
draw.line((24, filters_y1 + 8, 1416, filters_y1 + 8), fill=(230, 233, 236), width=1)

# Event card 1 background (rounded card behind image + content)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1175
card1_radius = 20
# shadow (simple offset rectangle to simulate shadow)
shadow_offset = 8
draw.rounded_rectangle(
    (card1_x + shadow_offset, card1_y + shadow_offset, card1_x + card1_w + shadow_offset, card1_y + card1_h + shadow_offset),
    radius=card1_radius + 2,
    fill=(235, 238, 240)
)
# main card
draw.rounded_rectangle(
    (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h),
    radius=card1_radius,
    fill=(255, 255, 255)
)
# inner subtle divider in the card (separating image area from details area)
# approximate location: image occupies top portion; draw a faint divider below where image will be pasted
image_split_y = card1_y + int(card1_h * 0.42)
draw.line((card1_x + 20, image_split_y, card1_x + card1_w - 20, image_split_y), fill=(245, 246, 247), width=1)

# Event card 2 background (rounded card behind next event)
card2_x, card2_y = 48, 1899
card2_w, card2_h = 1344, 917
card2_radius = 20
# shadow
draw.rounded_rectangle(
    (card2_x + shadow_offset, card2_y + shadow_offset, card2_x + card2_w + shadow_offset, card2_y + card2_h + shadow_offset),
    radius=card2_radius + 2,
    fill=(235, 238, 240)
)
# main card
draw.rounded_rectangle(
    (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h),
    radius=card2_radius,
    fill=(255, 255, 255)
)
# divider inside card 2 (approx where image ends)
image2_split_y = card2_y + int(card2_h * 0.60)
draw.line((card2_x + 20, image2_split_y, card2_x + card2_w - 20, image2_split_y), fill=(245, 246, 247), width=1)

# Thin separators between major sections
draw.line((24, card1_y - 24, 1416, card1_y - 24), fill=(240, 241, 243), width=1)
draw.line((24, card2_y - 24, 1416, card2_y - 24), fill=(240, 241, 243), width=1)

# Bottom navigation bar background (leave icons to be pasted)
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
# top divider for nav bar
draw.line((0, nav_top, 1440, nav_top), fill=(230, 233, 236), width=2)
# subtle shadow above nav bar
draw.line((0, nav_top - 6, 1440, nav_top - 6), fill=(245, 246, 247), width=1)

# Additional subtle UI divider near top content count area (where "274 events" will be pasted)
events_count_area_y = 320
draw.line((48, events_count_area_y + 64, 1392, events_count_area_y + 64), fill=(240, 241, 243), width=1)

# End of background and structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 432, 103)
    canvas.paste(_c0, (438, 410), _c0)
except Exception:
    pass
layout["Tomorrow"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (882, 410), _c1)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/03_icon_Business.png
try:
    _c3 = get_crop(3, 241, 103)
    canvas.paste(_c3, (1081, 410), _c3)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/04_icon_Business.png
try:
    _c4 = get_crop(4, 93, 110)
    canvas.paste(_c4, (1328, 407), _c4)
except Exception:
    pass
layout["Business"] = [1328, 407, 1421, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/05_icon_Close_current_screen.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1248, 96), _c5)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/06_icon_Manthly_Contribution_USD4.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Manthly_Contribution;_USD"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 65)
    canvas.paste(_c7, (1154, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1154, 0, 1202, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1092, 2415), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 2415), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 97, 65)
    canvas.paste(_c10, (1213, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1213, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 62, 61)
    canvas.paste(_c11, (309, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [309, 1, 371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/12_icon_8.08.png
try:
    _c12 = get_crop(12, 128, 121)
    canvas.paste(_c12, (53, 111), _c12)
except Exception:
    pass
layout["8.08"] = [53, 111, 181, 232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/13_icon_8.08.png
try:
    _c13 = get_crop(13, 56, 63)
    canvas.paste(_c13, (182, 1), _c13)
except Exception:
    pass
layout["8.08"] = [182, 1, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/14_icon_8.08.png
try:
    _c14 = get_crop(14, 57, 65)
    canvas.paste(_c14, (116, 0), _c14)
except Exception:
    pass
layout["8.08"] = [116, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 62)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 1, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 55, 64)
    canvas.paste(_c16, (1318, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1318, 0, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/17_icon_inner_journey_to_The_Sacred_Temple_of_yo.png
try:
    _c17 = get_crop(17, 1344, 1175)
    canvas.paste(_c17, (48, 676), _c17)
except Exception:
    pass
layout["inner_journey_to_The_Sacr"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/18_icon_Ticket_sales_end_soon.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/19_icon_community_events.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["community_events"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/20_icon_Spiritual_Leadership_Develpoment_YOU_ARE.png
try:
    _c20 = get_crop(20, 1344, 1175)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Spiritual_Leadership_Deve"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1236, 1192), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/22_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c22 = get_crop(22, 1344, 917)
    canvas.paste(_c22, (48, 1899), _c22)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 46, 62)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [384, 2, 430, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/24_icon_Chicago.png
try:
    _c24 = get_crop(24, 417, 144)
    canvas.paste(_c24, (0, 259), _c24)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/25_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/26_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/28_icon_Promoted.png
try:
    _c28 = get_crop(28, 245, 63)
    canvas.paste(_c28, (82, 1744), _c28)
except Exception:
    pass
layout["Promoted"] = [82, 1744, 327, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/29_icon_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/30_text_8.08.png
try:
    _c30 = get_crop(30, 91, 43)
    canvas.paste(_c30, (20, 17), _c30)
except Exception:
    pass
layout["8.08"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/31_text_274events.png
try:
    _c31 = get_crop(31, 372, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["274events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/32_text_Online.png
try:
    _c32 = get_crop(32, 129, 45)
    canvas.paste(_c32, (91, 1687), _c32)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_12_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-14/33_text_elem_33.png
try:
    _c33 = get_crop(33, 89, 30)
    canvas.paste(_c33, (104, 2779), _c33)
except Exception:
    pass
layout["_+"] = [104, 2779, 193, 2809]
