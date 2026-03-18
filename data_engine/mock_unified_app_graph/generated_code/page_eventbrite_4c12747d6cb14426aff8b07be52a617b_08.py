# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_08
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10.png
# step_index: 8/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
bg_color = (249, 250, 251)  # very light off-white
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top area)
status_h = 96
status_color = (200, 200, 200)  # light grey status bar
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Header / toolbar area below status bar
header_top = status_h
header_h = 128
header_color = (255, 255, 255)  # white header
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill=header_color)

# Divider line under header
divider_y = header_top + header_h
draw.line([(40, divider_y), (1400, divider_y)], fill=(220, 223, 226), width=2)

# Filter chips row background strip (subtle)
filters_top = divider_y + 24
filters_h = 120
draw.rectangle([(0, filters_top), (1440, filters_top + filters_h)], fill=(249, 250, 251))

# Thin separator under filters
draw.line([(48, filters_top + filters_h + 8), (1392, filters_top + filters_h + 8)], fill=(230, 233, 236), width=1)

# First event card container (rounded card with subtle shadow)
card_margin_x = 48
card1_top = filters_top + filters_h + 48
card1_width = 1344
card1_height = 420
card1_bbox = (card_margin_x, card1_top, card_margin_x + card1_width, card1_top + card1_height)

# shadow
shadow_offset = 8
shadow_color = (230, 233, 236)
draw.rounded_rectangle(
    [card1_bbox[0] + shadow_offset, card1_bbox[1] + shadow_offset,
     card1_bbox[2] + shadow_offset, card1_bbox[3] + shadow_offset],
    radius=20, fill=shadow_color
)
# card
card_color = (255, 255, 255)
draw.rounded_rectangle(list(card1_bbox), radius=20, fill=card_color)

# subtle inner divider for image/title area (leave space for pasted image)
img_divider_y = card1_top + 300
draw.line([(card1_bbox[0] + 24, img_divider_y), (card1_bbox[2] - 24, img_divider_y)], fill=(242, 244, 246), width=1)

# Space between cards
between_space = 60

# Second event card container (further down)
card2_top = card1_top + card1_height + between_space
card2_height = 420
card2_bbox = (card_margin_x, card2_top, card_margin_x + card1_width, card2_top + card2_height)

# shadow for second card
draw.rounded_rectangle(
    [card2_bbox[0] + shadow_offset, card2_bbox[1] + shadow_offset,
     card2_bbox[2] + shadow_offset, card2_bbox[3] + shadow_offset],
    radius=20, fill=shadow_color
)
# card body
draw.rounded_rectangle(list(card2_bbox), radius=20, fill=card_color)

# inner divider for second card
img2_divider_y = card2_top + 300
draw.line([(card2_bbox[0] + 24, img2_divider_y), (card2_bbox[2] - 24, img2_divider_y)], fill=(242, 244, 246), width=1)

# Light section background for a large content area below (to hint separation)
content_section_top = card2_top + card2_height + 48
draw.rectangle([(0, content_section_top), (1440, content_section_top + 240)], fill=(247, 248, 250))

# Horizontal separators between main sections
sep_positions = [
    card1_top - 24,
    card1_top + card1_height + 12,
    card2_top + card2_height + 12,
    content_section_top + 240
]
for y in sep_positions:
    draw.line([(40, y), (1400, y)], fill=(235, 238, 241), width=1)

# Bottom navigation bar (safe area)
nav_h = 120
nav_top = canvas.size[1] - nav_h
# top border shadow
draw.rectangle([(0, nav_top), (1440, nav_top + nav_h)], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 228, 231), width=2)

# subtle inner highlight on nav
draw.line([(0, nav_top + 2), (1440, nav_top + 2)], fill=(255, 255, 255), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/05_icon_of_he_month_at_8_PM.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2288), _c5)
except Exception:
    pass
layout["of_#he_month_at_8_PM"] = [1092, 2288, 1236, 2432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/06_icon_Fo.png
try:
    _c6 = get_crop(6, 137, 110)
    canvas.paste(_c6, (1295, 406), _c6)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2288), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2288, 1380, 2432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/09_icon_7.52.png
try:
    _c9 = get_crop(9, 119, 110)
    canvas.paste(_c9, (57, 116), _c9)
except Exception:
    pass
layout["7.52"] = [57, 116, 176, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/11_icon_Bad_Art_Creative_Misery_Loves_Comedy.png
try:
    _c11 = get_crop(11, 1344, 945)
    canvas.paste(_c11, (48, 1772), _c11)
except Exception:
    pass
layout["Bad_Art:_Creative_Misery_"] = [48, 1772, 1392, 2717]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/12_icon_7.52.png
try:
    _c12 = get_crop(12, 60, 64)
    canvas.paste(_c12, (181, 0), _c12)
except Exception:
    pass
layout["7.52"] = [181, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/13_icon_Art.png
try:
    _c13 = get_crop(13, 67, 62)
    canvas.paste(_c13, (308, 1), _c13)
except Exception:
    pass
layout["Art"] = [308, 1, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/14_icon_7.52.png
try:
    _c14 = get_crop(14, 61, 65)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["7.52"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/15_icon_Art.png
try:
    _c15 = get_crop(15, 54, 63)
    canvas.paste(_c15, (246, 1), _c15)
except Exception:
    pass
layout["Art"] = [246, 1, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 57, 61)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1376, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 90, 61)
    canvas.paste(_c17, (1207, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1207, 0, 1297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/18_icon_WED_APRIL_24.png
try:
    _c18 = get_crop(18, 1344, 1048)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["WED_APRIL_24"] = [48, 676, 1392, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/19_icon_San_Francisco.png
try:
    _c19 = get_crop(19, 536, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 52, 61)
    canvas.paste(_c20, (383, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/21_icon_Fri_May_17_._8.00_PM_PDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Fri,_May_17_._8.00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/22_icon_7.52.png
try:
    _c22 = get_crop(22, 142, 64)
    canvas.paste(_c22, (11, 0), _c22)
except Exception:
    pass
layout["7.52"] = [11, 0, 153, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/23_icon_AII_Out_Comedy_Theater-Improv_Classes_an.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["AII_Out_Comedy_Theater-Im"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 41, 62)
    canvas.paste(_c24, (1273, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1273, 0, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/25_icon_Fri_May_17_._8.00_PM_PDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Fri,_May_17_._8.00_PM_PDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/27_text_Art.png
try:
    _c27 = get_crop(27, 123, 73)
    canvas.paste(_c27, (203, 135), _c27)
except Exception:
    pass
layout["Art"] = [203, 135, 326, 208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/28_text_1_324events.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["1,324events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/29_text_At.png
try:
    _c29 = get_crop(29, 162, 86)
    canvas.paste(_c29, (647, 693), _c29)
except Exception:
    pass
layout["At"] = [647, 693, 809, 779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/30_text_68270_Niott.png
try:
    _c30 = get_crop(30, 251, 54)
    canvas.paste(_c30, (606, 833), _c30)
except Exception:
    pass
layout["68270_Niott"] = [606, 833, 857, 887]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/31_clickable_Art.png
try:
    _c31 = get_crop(31, 1344, 191)
    canvas.paste(_c31, (48, 72), _c31)
except Exception:
    pass
layout["Art"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_08_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-10/32_clickable_Favorites.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (576, 2804), _c32)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
