# page_id: page_eventbrite_e381830686d842d08e553d1397c2110d_03
# screenshot: 2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5.png
# step_index: 3/3
# task: Open Eventbrite. Open "Recommended". Select the third recommended event. Add it to favourites. What is the refund policy?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite mobile page
# Assumes: canvas (1440x2960 RGB PIL.Image) and draw (PIL.ImageDraw.Draw) exist.
# Provided fonts: font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors (approximate to screenshot)
status_bar_color = (220, 220, 220)    # light gray status bar
hero_bg = (75, 86, 92)                # desaturated dark for hero/banner background
hero_stripe = (55, 121, 128)          # teal stripe at bottom of hero image
card_bg = (249, 250, 252)             # very light card background
card_border = (233, 235, 240)         # subtle card border
separator = (234, 235, 237)           # thin separators
ticket_border = (57, 92, 255)         # bright blue ticket border
page_bg = (255, 255, 255)             # page background (white)

# Fill page background (canvas already white; rewrite to ensure consistent)
draw.rectangle([0, 0, w, h], fill=page_bg)

# 1) Status bar area at top (~56 px)
status_h = 56
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# 2) Hero/banner area (image placeholder background)
hero_top = status_h
hero_bottom = 420
draw.rectangle([0, hero_top, w, hero_bottom], fill=hero_bg)

# Add a slanted teal stripe across the bottom of the hero to mimic design accent
# Create a simple trapezoid/polygon for the stripe
stripe_height = 72
stripe_y0 = hero_bottom - stripe_height
# Slight diagonal: left lower, right slightly higher
stripe_poly = [(0, hero_bottom), (0, stripe_y0 + 10), (w, stripe_y0 - 24), (w, hero_bottom)]
draw.polygon(stripe_poly, fill=hero_stripe)

# Slight drop shadow line under hero
shadow_y = hero_bottom + 6
draw.rectangle([0, hero_bottom, w, shadow_y], fill=(240, 240, 240))

# 3) Organizer / follow card (rounded)
card_x0 = 48
card_x1 = w - 48
card_y0 = 540
card_y1 = 720
card_radius = 28
# Draw card background with subtle border
try:
    draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=card_radius, fill=card_bg, outline=card_border, width=2)
except Exception:
    # Fallback if rounded_rectangle is unavailable: draw rectangle
    draw.rectangle([card_x0, card_y0, card_x1, card_y1], fill=card_bg, outline=card_border)

# Subtle inner divider inside the organizer card (to visually separate avatar area from follow button)
divider_x = card_x1 - 300
draw.line([(divider_x, card_y0 + 18), (divider_x, card_y1 - 18)], fill=(245,245,247), width=1)

# 4) Thin separators between sections
# Under the refund text area (approximate)
sep1_y = 1280
draw.line([(48, sep1_y), (w - 48, sep1_y)], fill=separator, width=2)

# Between content and ticket selection area
sep2_y = 2320
draw.line([(24, sep2_y), (w - 24, sep2_y)], fill=separator, width=2)

# A lighter divider closer to the "About this event" area
sep3_y = 1820
draw.line([(48, sep3_y), (w - 48, sep3_y)], fill=separator, width=1)

# 5) Ticket selection box (rounded rectangle with blue border)
ticket_x0 = 40
ticket_x1 = w - 40
ticket_y0 = 2360
ticket_y1 = 2690
ticket_radius = 22
border_width = 8

# Draw outer border by drawing a thicker rounded rectangle then inner fill
# Outer border
try:
    draw.rounded_rectangle([ticket_x0, ticket_y0, ticket_x1, ticket_y1], radius=ticket_radius, fill=None, outline=ticket_border, width=border_width)
    # Inner white fill (inset by half the border width)
    inset = border_width // 2 + 2
    draw.rounded_rectangle([ticket_x0 + inset, ticket_y0 + inset, ticket_x1 - inset, ticket_y1 - inset],
                           radius=max(0, ticket_radius - inset), fill=(255,255,255), outline=None)
except Exception:
    # Fallback simple rectangle border + fill
    draw.rectangle([ticket_x0, ticket_y0, ticket_x1, ticket_y1], fill=(255,255,255), outline=ticket_border, width=4)

# Subtle horizontal rule inside ticket box (to separate title from price area)
inner_sep_y = ticket_y0 + 84
draw.line([(ticket_x0 + 30, inner_sep_y), (ticket_x1 - 30, inner_sep_y)], fill=(245,245,247), width=1)

# 6) Page left/right margins: faint vertical guide lines (purely structural, very subtle)
margin_x = 48
draw.line([(margin_x, hero_bottom), (margin_x, h)], fill=(250,250,250), width=1)
draw.line([(w - margin_x, hero_bottom), (w - margin_x, h)], fill=(250,250,250), width=1)

# 7) Additional subtle bottom area divider above reserve button
reserve_button_top = 2756  # (detected element pasted later)
draw.line([(24, reserve_button_top - 18), (w - 24, reserve_button_top - 18)], fill=(238, 238, 238), width=1)

# 8) Accessibility: Draw faint focus/background blocks for "About this event" section header area
about_block_y0 = 1600
about_block_y1 = about_block_y0 + 240
draw.rectangle([48, about_block_y0, w - 48, about_block_y1], fill=(255,255,255))

# Done - background, structural cards, separators and ticket box.
# The detected icons, text, and buttons will be pasted on top of these structures.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/02_icon_Increase.png
try:
    _c2 = get_crop(2, 96, 96)
    canvas.paste(_c2, (1224, 2444), _c2)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/03_icon_7.02.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["7.02"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 93, 103)
    canvas.paste(_c4, (1108, 2441), _c4)
except Exception:
    pass
layout["icon_4"] = [1108, 2441, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/05_icon_Reserve_a_spot.png
try:
    _c5 = get_crop(5, 1296, 132)
    canvas.paste(_c5, (72, 2756), _c5)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/06_icon_Understanding.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["Understanding"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/07_icon_7.02.png
try:
    _c7 = get_crop(7, 64, 64)
    canvas.paste(_c7, (179, 2), _c7)
except Exception:
    pass
layout["7.02"] = [179, 2, 243, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/08_icon_grief.png
try:
    _c8 = get_crop(8, 234, 144)
    canvas.paste(_c8, (48, 2090), _c8)
except Exception:
    pass
layout["grief"] = [48, 2090, 282, 2234]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/09_icon_7.02.png
try:
    _c9 = get_crop(9, 59, 64)
    canvas.paste(_c9, (116, 1), _c9)
except Exception:
    pass
layout["7.02"] = [116, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/10_icon_Share.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1260, 108), _c10)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 44, 59)
    canvas.paste(_c11, (1327, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [1327, 4, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 60)
    canvas.paste(_c12, (247, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 3, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 43, 57)
    canvas.paste(_c13, (1271, 6), _c13)
except Exception:
    pass
layout["icon_13"] = [1271, 6, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 58)
    canvas.paste(_c14, (1216, 4), _c14)
except Exception:
    pass
layout["icon_14"] = [1216, 4, 1282, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 61, 61)
    canvas.paste(_c15, (311, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [311, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 64)
    canvas.paste(_c16, (382, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 2, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/17_icon_Free.png
try:
    _c17 = get_crop(17, 135, 100)
    canvas.paste(_c17, (100, 2578), _c17)
except Exception:
    pass
layout["Free"] = [100, 2578, 235, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/18_icon_Free.png
try:
    _c18 = get_crop(18, 75, 72)
    canvas.paste(_c18, (249, 2588), _c18)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/19_text_7.02.png
try:
    _c19 = get_crop(19, 89, 43)
    canvas.paste(_c19, (22, 17), _c19)
except Exception:
    pass
layout["7.02"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/20_text_Grief_and_Loss.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1116, 108), _c20)
except Exception:
    pass
layout["Grief_and_Loss"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/21_text_Institute.png
try:
    _c21 = get_crop(21, 125, 30)
    canvas.paste(_c21, (233, 566), _c21)
except Exception:
    pass
layout["Institute"] = [233, 566, 358, 596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/22_text_Wednesday_June_26.png
try:
    _c22 = get_crop(22, 379, 144)
    canvas.paste(_c22, (288, 1028), _c22)
except Exception:
    pass
layout["Wednesday,_June_26"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/23_text_1O_00_AM.png
try:
    _c23 = get_crop(23, 239, 54)
    canvas.paste(_c23, (585, 766), _c23)
except Exception:
    pass
layout["1O:00_AM"] = [585, 766, 824, 820]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/24_text_Understanding_Grief_and_Loss.png
try:
    _c24 = get_crop(24, 379, 144)
    canvas.paste(_c24, (288, 1028), _c24)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/25_text_Instit.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (96, 1067), _c25)
except Exception:
    pass
layout["Instit"] = [96, 1067, 240, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/26_text_Institute_on.png
try:
    _c26 = get_crop(26, 379, 144)
    canvas.paste(_c26, (288, 1028), _c26)
except Exception:
    pass
layout["Institute_on"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/27_text_1.3k_Followers.png
try:
    _c27 = get_crop(27, 379, 144)
    canvas.paste(_c27, (288, 1028), _c27)
except Exception:
    pass
layout["1.3k_Followers"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/28_text_Online_event.png
try:
    _c28 = get_crop(28, 274, 55)
    canvas.paste(_c28, (139, 1341), _c28)
except Exception:
    pass
layout["Online_event"] = [139, 1341, 413, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/29_text_hrs.png
try:
    _c29 = get_crop(29, 77, 50)
    canvas.paste(_c29, (176, 1452), _c29)
except Exception:
    pass
layout["hrs"] = [176, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/30_text_Refund_policy.png
try:
    _c30 = get_crop(30, 299, 63)
    canvas.paste(_c30, (138, 1558), _c30)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/31_text_The_organizer_will_review_refund_request.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1295), _c31)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_03_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-5/32_text_General_Admission.png
try:
    _c32 = get_crop(32, 75, 72)
    canvas.paste(_c32, (249, 2588), _c32)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
