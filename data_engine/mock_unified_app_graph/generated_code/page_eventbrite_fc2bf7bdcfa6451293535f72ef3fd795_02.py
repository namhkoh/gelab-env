# page_id: page_eventbrite_fc2bf7bdcfa6451293535f72ef3fd795_02
# screenshot: 2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4.png
# step_index: 2/8
# task: Open Eventbrite. Search for events by 'Music' under online events. Choose the second event in the list. Get the event's duration information.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure drawing for mobile UI mockup
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_fill = (247, 249, 251)        # very light cool background
status_bar_color = (158, 158, 158)   # top status bar gray
header_bg = (255, 255, 255)      # white header/search background
divider_color = (224, 227, 230)  # subtle divider
card_shadow = (230, 235, 240)    # soft shadow behind cards
image_placeholder_1 = (204, 232, 255)  # pale blue for image placeholder
image_placeholder_2 = (255, 244, 230)  # pale warm for second image placeholder
bottom_bar_bg = (255, 255, 255)  # bottom nav background
subtle_grey = (245, 246, 248)

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_fill)

# Status bar area (~72px)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header / search background (under status bar)
search_top = status_h
search_bottom = 263  # leave space for search field area (do not draw its contents)
draw.rectangle([(0, search_top), (W, search_bottom)], fill=header_bg)

# Thin divider under the search area
draw.line([(24, search_bottom), (W-24, search_bottom)], fill=divider_color, width=2)

# Divider above chips/filter row (approx where chips start)
chips_y = 410
draw.line([(24, chips_y), (W-24, chips_y)], fill=divider_color, width=1)

# First event card (image + card background)
card1_x = 48
card1_y = 676
card1_w = 1344
card1_h = 1096
card1_bbox = (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h)
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    [card1_bbox[0] + shadow_offset, card1_bbox[1] + shadow_offset,
     card1_bbox[2] + shadow_offset, card1_bbox[3] + shadow_offset],
    radius=28, fill=card_shadow)
# image placeholder rounded rect
draw.rounded_rectangle(card1_bbox, radius=28, fill=image_placeholder_1)

# White area directly below first card for title/date background (keeps separation)
text_bg_top = card1_bbox[3]
text_bg_bottom = 1820  # stop just above second event start (so spacing matches)
if text_bg_bottom > text_bg_top:
    draw.rectangle([(24, text_bg_top), (W-24, text_bg_bottom)], fill=header_bg)

# subtle separator before next card
sep_y = text_bg_bottom
draw.line([(24, sep_y), (W-24, sep_y)], fill=divider_color, width=1)

# Second event card (wide banner style)
card2_x = 48
card2_y = 1820
card2_w = 1344
card2_h = 996
card2_bbox = (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h)
# shadow for second card
draw.rounded_rectangle(
    [card2_bbox[0] + shadow_offset, card2_bbox[1] + shadow_offset,
     card2_bbox[2] + shadow_offset, card2_bbox[3] + shadow_offset],
    radius=22, fill=card_shadow)
# image/banner placeholder
draw.rounded_rectangle(card2_bbox, radius=22, fill=image_placeholder_2)

# Divider line between content area and bottom region (above nav)
bottom_div_y = 2816
draw.line([(0, bottom_div_y), (W, bottom_div_y)], fill=divider_color, width=1)

# Bottom navigation bar background area (~120px high)
bottom_h = 120
bottom_top = H - bottom_h
draw.rectangle([(0, bottom_top), (W, H)], fill=bottom_bar_bg)
# top border for bottom bar
draw.line([(0, bottom_top), (W, bottom_top)], fill=divider_color, width=2)

# Small subtle horizontal rule near top of page (under location row)
loc_div_y = 259
draw.line([(24, loc_div_y), (W-24, loc_div_y)], fill=subtle_grey, width=1)

# Add faint inner separators between major sections (subtle)
draw.line([(24, 480), (W-24, 480)], fill=subtle_grey, width=1)
draw.line([(24, 1160), (W-24, 1160)], fill=subtle_grey, width=1)

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/04_icon_Ghibli_Music_Night.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Ghibli_Music_Night"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/05_icon_Foo.png
try:
    _c5 = get_crop(5, 149, 110)
    canvas.paste(_c5, (1283, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/06_icon_Lnttl.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2336), _c6)
except Exception:
    pass
layout["Lnttl"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2336), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 1344, 191)
    canvas.paste(_c9, (48, 72), _c9)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/10_icon_8.04.png
try:
    _c10 = get_crop(10, 117, 108)
    canvas.paste(_c10, (59, 117), _c10)
except Exception:
    pass
layout["8.04"] = [59, 117, 176, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/11_icon_8.04.png
try:
    _c11 = get_crop(11, 60, 63)
    canvas.paste(_c11, (181, 0), _c11)
except Exception:
    pass
layout["8.04"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 69, 63)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/13_icon_8.04.png
try:
    _c13 = get_crop(13, 61, 65)
    canvas.paste(_c13, (114, 0), _c13)
except Exception:
    pass
layout["8.04"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 62)
    canvas.paste(_c14, (248, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [248, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 62, 59)
    canvas.paste(_c15, (1316, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1316, 0, 1378, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/16_icon_Los_Angeles.png
try:
    _c16 = get_crop(16, 492, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 75, 60)
    canvas.paste(_c17, (1208, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1208, 0, 1283, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/18_icon_Ghibli_Music_Night.png
try:
    _c18 = get_crop(18, 1344, 1096)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["Ghibli_Music_Night"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 52, 61)
    canvas.paste(_c19, (383, 2), _c19)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/20_icon_Promoted.png
try:
    _c20 = get_crop(20, 244, 65)
    canvas.paste(_c20, (85, 1665), _c20)
except Exception:
    pass
layout["Promoted"] = [85, 1665, 329, 1730]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/21_icon_NeueHouse_Hollywood.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["NeueHouse_Hollywood"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/22_icon_6.30_PM_PDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["6.30_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/23_icon_SLallr_Tande_Couini_hItnat_Pir_Un_4_7_9n.png
try:
    _c23 = get_crop(23, 1344, 996)
    canvas.paste(_c23, (48, 1820), _c23)
except Exception:
    pass
layout["SLallr_Tande{Couini_hItna"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/24_icon_Lnttl.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Lnttl"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/25_icon_Lnttl.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["Lnttl"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 40, 60)
    canvas.paste(_c26, (1274, 0), _c26)
except Exception:
    pass
layout["icon_26"] = [1274, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/27_icon_6.30_PM_PDT.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["6.30_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/28_icon_Mon_Apr_29.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Mon,_Apr_29"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/29_text_8.04.png
try:
    _c29 = get_crop(29, 94, 45)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["8.04"] = [20, 15, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/31_text_8898.png
try:
    _c31 = get_crop(31, 280, 102)
    canvas.paste(_c31, (1085, 685), _c31)
except Exception:
    pass
layout["8898"] = [1085, 685, 1365, 787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/32_text_Jordan_High_School_Atlantic_Avenue_Long_.png
try:
    _c32 = get_crop(32, 1344, 1096)
    canvas.paste(_c32, (48, 676), _c32)
except Exception:
    pass
layout["Jordan_High_School,_Atlan"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/33_text_SECOUD_ANNUAL_BENEFLT_GALlA.png
try:
    _c33 = get_crop(33, 1344, 996)
    canvas.paste(_c33, (48, 1820), _c33)
except Exception:
    pass
layout["SECOUD_ANNUAL_BENEFLT_GAL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/34_text_Ax.png
try:
    _c34 = get_crop(34, 69, 41)
    canvas.paste(_c34, (475, 1990), _c34)
except Exception:
    pass
layout["Ax"] = [475, 1990, 544, 2031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/35_text_Mon_Apr_29.png
try:
    _c35 = get_crop(35, 244, 54)
    canvas.paste(_c35, (93, 2678), _c35)
except Exception:
    pass
layout["Mon,_Apr_29"] = [93, 2678, 337, 2732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/36_text_6.30_PM_PDT.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (288, 2804), _c36)
except Exception:
    pass
layout["6.30_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_02_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-4/37_text_NeueHouse_Hollywood.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (288, 2804), _c37)
except Exception:
    pass
layout["NeueHouse_Hollywood"] = [288, 2804, 576, 2960]
