# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_15
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17.png
# step_index: 15/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Draw the background and structural elements of the UI.

w, h = canvas.size

# Colors
bg_color = (250, 250, 252)         # very light off-white background
status_bar_color = (158, 158, 158) # top status bar gray
header_bg = (255, 255, 255)        # white header area
divider_color = (224, 224, 230)    # subtle light divider
card_shadow = (230, 232, 238)      # soft shadow for cards
card_bg = (255, 255, 255)          # card background (white)
nav_bg = (255, 255, 255)           # bottom nav background
muted_line = (235, 235, 240)       # separators

# Fill full background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (~72px high)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header/Search area
header_top = status_h
header_bottom = 280
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Thin divider under header
draw.line([(48, header_bottom - 8), (w - 48, header_bottom - 8)], fill=divider_color, width=2)

# Secondary thin separator under filters area (approx where chips and event count sit)
sep_y = 460
draw.line([(24, sep_y), (w - 24, sep_y)], fill=muted_line, width=1)

# Card 1 container (rounded) behind first event card/image
card_margin_x = 48
card1_top = 620
card1_bottom = 1760
card_radius = 28

# Shadow for card1 (offset)
draw.rounded_rectangle(
    [(card_margin_x + 8, card1_top + 10), (w - card_margin_x + 8, card1_bottom + 10)],
    radius=card_radius + 2,
    fill=card_shadow
)

# Main card1 background
draw.rounded_rectangle(
    [(card_margin_x, card1_top), (w - card_margin_x, card1_bottom)],
    radius=card_radius,
    fill=card_bg,
    outline=divider_color
)

# Thin divider separating image area and text area within card1 (approx)
inner_div_y = card1_top + 540
draw.line([(card_margin_x + 16, inner_div_y), (w - card_margin_x - 16, inner_div_y)], fill=muted_line, width=1)

# Card 2 container (rounded) behind second event card/image
card2_top = card1_bottom + 36
card2_bottom = 2600
draw.rounded_rectangle(
    [(card_margin_x + 8, card2_top + 10), (w - card_margin_x + 8, card2_bottom + 10)],
    radius=card_radius + 2,
    fill=card_shadow
)
draw.rounded_rectangle(
    [(card_margin_x, card2_top), (w - card_margin_x, card2_bottom)],
    radius=card_radius,
    fill=card_bg,
    outline=divider_color
)

# Subtle separator lines between sections/cards
draw.line([(24, card1_bottom + 22), (w - 24, card1_bottom + 22)], fill=muted_line, width=1)
draw.line([(24, card2_bottom + 8), (w - 24, card2_bottom + 8)], fill=muted_line, width=1)

# Bottom navigation bar background and top divider
nav_h = 160
nav_top = h - nav_h
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)
draw.line([(0, nav_top), (w, nav_top)], fill=divider_color, width=2)

# Small rounded "floating" white background for the main list area near top (subtle)
floating_y1 = header_bottom + 8
floating_y2 = floating_y1 + 60
draw.rounded_rectangle([(36, floating_y1), (w - 36, floating_y2)], radius=18, fill=(255,255,255), outline=divider_color)

# Additional subtle vertical padding guides (visual structure only)
left_padding_x = 48
right_padding_x = w - 48
draw.line([(left_padding_x, header_bottom), (left_padding_x, card2_bottom)], fill=(245,245,248), width=1)
draw.line([(right_padding_x, header_bottom), (right_padding_x, card2_bottom)], fill=(245,245,248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1034, 410), _c0)
except Exception:
    pass
layout["Music"] = [1034, 410, 1221, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/01_icon_15_2024.png
try:
    _c1 = get_crop(1, 584, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["15,_2024"] = [438, 410, 1022, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2415), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2415), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/05_icon_mictt.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["mictt"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/06_icon_Busine.png
try:
    _c6 = get_crop(6, 159, 103)
    canvas.paste(_c6, (1233, 410), _c6)
except Exception:
    pass
layout["Busine"] = [1233, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/07_icon_mictt.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["mictt"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/08_icon_9pM.png
try:
    _c8 = get_crop(8, 1344, 1175)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["[9pM:"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 66)
    canvas.paste(_c9, (1152, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1152, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/10_icon_7.48.png
try:
    _c10 = get_crop(10, 122, 114)
    canvas.paste(_c10, (56, 114), _c10)
except Exception:
    pass
layout["7.48"] = [56, 114, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/11_icon_7.48.png
try:
    _c11 = get_crop(11, 62, 65)
    canvas.paste(_c11, (179, 0), _c11)
except Exception:
    pass
layout["7.48"] = [179, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 68, 64)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 87, 63)
    canvas.paste(_c13, (1213, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1213, 0, 1300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/14_icon_7.48.png
try:
    _c14 = get_crop(14, 61, 65)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["7.48"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 65)
    canvas.paste(_c15, (246, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [246, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 56, 61)
    canvas.paste(_c16, (1317, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1317, 0, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/17_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c17 = get_crop(17, 1344, 917)
    canvas.paste(_c17, (48, 1899), _c17)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/18_icon_Path_to_Ownershin.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Path_to_Ownershin"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/19_icon_Free.png
try:
    _c19 = get_crop(19, 127, 78)
    canvas.paste(_c19, (91, 2592), _c19)
except Exception:
    pass
layout["Free"] = [91, 2592, 218, 2670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/20_icon_Search_forae.png
try:
    _c20 = get_crop(20, 50, 64)
    canvas.paste(_c20, (383, 1), _c20)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/21_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 1344, 191)
    canvas.paste(_c22, (48, 72), _c22)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/23_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/24_icon_Path_to_Ownershin.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Path_to_Ownershin"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/25_icon_San_Francisco.png
try:
    _c25 = get_crop(25, 536, 144)
    canvas.paste(_c25, (0, 259), _c25)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 245, 61)
    canvas.paste(_c26, (83, 1746), _c26)
except Exception:
    pass
layout["Promoted"] = [83, 1746, 328, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/27_icon_7.48.png
try:
    _c27 = get_crop(27, 98, 64)
    canvas.paste(_c27, (9, 0), _c27)
except Exception:
    pass
layout["7.48"] = [9, 0, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/28_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (576, 2804), _c28)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/29_icon_Super_Bass_Hip_Hop_Thursdays_Party_at_Be.png
try:
    _c29 = get_crop(29, 1344, 1175)
    canvas.paste(_c29, (48, 676), _c29)
except Exception:
    pass
layout["Super_Bass_Hip_Hop_Thursd"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 62)
    canvas.paste(_c30, (1273, 0), _c30)
except Exception:
    pass
layout["icon_30"] = [1273, 0, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/31_text_6_472_events.png
try:
    _c31 = get_crop(31, 372, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["6,472_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_15_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-17/32_text_Beaux.png
try:
    _c32 = get_crop(32, 124, 43)
    canvas.paste(_c32, (94, 1689), _c32)
except Exception:
    pass
layout["Beaux"] = [94, 1689, 218, 1732]
