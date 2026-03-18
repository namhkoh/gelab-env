# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_10
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12.png
# step_index: 10/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw UI background and structure for the Eventbrite-like page
# available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (189, 189, 189)      # light grey status bar
divider_color = (234, 232, 236)         # very light grey divider
header_shadow = (245, 244, 246)         # subtle shadow
card_outline = (234, 232, 236)          # card outline
card_bg = (255, 255, 255)               # white card background
light_bg = (250, 250, 252)              # slightly off-white background
blue_border = (51, 102, 255)            # bright blue for ticket selector border
reserve_orange = (201, 62, 28)          # Reserve button orange
shadow_gray = (220, 220, 220)           # generic shadow

W, H = canvas.size

# 1) overall background wash (very subtle off-white)
draw.rectangle([(0, 0), (W, H)], fill=light_bg)

# 2) status bar at the top (~56 px)
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# 3) header/toolbar area (below status bar)
header_y0 = status_h
header_y1 = 180
# keep header white but add a subtle top-to-bottom slight shadow line for separation
draw.rectangle([(0, header_y0), (W, header_y1)], fill=card_bg)
draw.line([(48, header_y1), (W-48, header_y1)], fill=divider_color, width=1)

# 4) horizontal separators for contact/report area
# There is a small divider above and below the "Contact the organizer / Report event" row
sep1_y = 720
sep2_y = 860
draw.line([(48, sep1_y), (W-48, sep1_y)], fill=divider_color, width=1)
draw.line([(48, sep2_y), (W-48, sep2_y)], fill=divider_color, width=1)

# 5) "More like this" list item cards backgrounds (rounded rectangles)
list_cards = [
    (48, 1187, 48 + 1344, 1187 + 387),
    (48, 1670, 48 + 1344, 1670 + 326),
    (48, 2092, 48 + 1344, 2092 + 232)
]
for (x0, y0, x1, y1) in list_cards:
    # subtle drop shadow (offset)
    shadow_offset = 6
    draw.rounded_rectangle(
        [(x0 + shadow_offset, y0 + shadow_offset), (x1 + shadow_offset, y1 + shadow_offset)],
        radius=14,
        fill=shadow_gray
    )
    # white card
    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)],
        radius=14,
        fill=card_bg,
        outline=card_outline,
        width=1
    )
    # separator line at bottom of card (very light)
    draw.line([(x0 + 12, y1 - 1), (x1 - 12, y1 - 1)], fill=divider_color, width=1)

# 6) thin separators between list sections (a bit more subtle)
draw.line([(48, 1568), (W-48, 1568)], fill=divider_color, width=1)  # between first and second card area
draw.line([(48, 1986), (W-48, 1986)], fill=divider_color, width=1)  # between second and third card area

# 7) ticket selector box above the reserve button
ticket_x0 = 48
ticket_x1 = W - 48
ticket_y0 = 2320
ticket_y1 = 2720
# shadow
draw.rounded_rectangle(
    [(ticket_x0 + 3, ticket_y0 + 6), (ticket_x1 + 3, ticket_y1 + 6)],
    radius=20,
    fill=shadow_gray
)
# main white panel
draw.rounded_rectangle(
    [(ticket_x0, ticket_y0), (ticket_x1, ticket_y1)],
    radius=20,
    fill=card_bg,
    outline=blue_border,
    width=6
)

# inner divider line inside ticket box (light)
inner_div_y = ticket_y0 + 80
draw.line([(ticket_x0 + 24, inner_div_y), (ticket_x1 - 24, inner_div_y)], fill=divider_color, width=1)

# 8) quantity control background placeholders (right side inside ticket box)
# draw subtle rounded squares where the minus/plus controls will be pasted (only backgrounds)
ctrl_w = 84
ctrl_h = 84
ctrl_gap = 24
ctrl_r = 14
# minus button background (left gray)
minus_x1 = ticket_x1 - ctrl_gap - ctrl_w*2 - 24
minus_y1 = ticket_y0 + (ticket_y1 - ticket_y0 - ctrl_h)//2
draw.rounded_rectangle(
    [(minus_x1, minus_y1), (minus_x1 + ctrl_w, minus_y1 + ctrl_h)],
    radius=ctrl_r,
    fill=(245,245,247)
)
# plus button background (blue)
plus_x1 = ticket_x1 - ctrl_gap - ctrl_w
plus_y1 = minus_y1
draw.rounded_rectangle(
    [(plus_x1, plus_y1), (plus_x1 + ctrl_w, plus_y1 + ctrl_h)],
    radius=ctrl_r,
    fill=blue_border
)

# 9) Reserve button at the bottom (rounded orange button)
reserve_x = 72
reserve_y = 2756
reserve_w = 1296
reserve_h = 132
reserve_radius = 10
# slight shadow behind button
draw.rounded_rectangle(
    [(reserve_x + 0, reserve_y + 6), (reserve_x + reserve_w + 0, reserve_y + reserve_h + 6)],
    radius=reserve_radius,
    fill=shadow_gray
)
draw.rounded_rectangle(
    [(reserve_x, reserve_y), (reserve_x + reserve_w, reserve_y + reserve_h)],
    radius=reserve_radius,
    fill=reserve_orange
)

# 10) subtle bottom safe-area line
draw.line([(0, H-1), (W, H-1)], fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/00_icon_basement.png
try:
    _c0 = get_crop(0, 259, 144)
    canvas.paste(_c0, (641, 317), _c0)
except Exception:
    pass
layout["basement"] = [641, 317, 900, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/01_icon_backpacking.png
try:
    _c1 = get_crop(1, 307, 144)
    canvas.paste(_c1, (286, 317), _c1)
except Exception:
    pass
layout["backpacking"] = [286, 317, 593, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/02_icon_clinic.png
try:
    _c2 = get_crop(2, 169, 144)
    canvas.paste(_c2, (948, 317), _c2)
except Exception:
    pass
layout["clinic"] = [948, 317, 1117, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/03_icon_sports.png
try:
    _c3 = get_crop(3, 190, 144)
    canvas.paste(_c3, (48, 317), _c3)
except Exception:
    pass
layout["sports"] = [48, 317, 238, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/04_icon_berkeley.png
try:
    _c4 = get_crop(4, 230, 144)
    canvas.paste(_c4, (48, 492), _c4)
except Exception:
    pass
layout["berkeley"] = [48, 492, 278, 636]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/05_icon_clinic.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["clinic"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/06_icon_Share.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 108), _c6)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/07_icon_Reserve_a_spot.png
try:
    _c7 = get_crop(7, 1296, 132)
    canvas.paste(_c7, (72, 2756), _c7)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/08_icon_4.36.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["4.36"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/09_icon_Decrease.png
try:
    _c9 = get_crop(9, 99, 96)
    canvas.paste(_c9, (996, 2444), _c9)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/10_icon_8_323_followers.png
try:
    _c10 = get_crop(10, 1344, 387)
    canvas.paste(_c10, (48, 1187), _c10)
except Exception:
    pass
layout["8_323_followers"] = [48, 1187, 1392, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/11_icon_Increase.png
try:
    _c11 = get_crop(11, 96, 96)
    canvas.paste(_c11, (1224, 2444), _c11)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 91, 102)
    canvas.paste(_c12, (1109, 2442), _c12)
except Exception:
    pass
layout["icon_12"] = [1109, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/13_icon_closed_onllne_therapeutic_art_group_t0.png
try:
    _c13 = get_crop(13, 1344, 232)
    canvas.paste(_c13, (48, 2092), _c13)
except Exception:
    pass
layout["closed_onllne_therapeutic"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/14_icon_Basement_Berkeleyl.png
try:
    _c14 = get_crop(14, 1344, 326)
    canvas.paste(_c14, (48, 1670), _c14)
except Exception:
    pass
layout["Basement_Berkeleyl"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/15_icon_Bs.png
try:
    _c15 = get_crop(15, 1344, 326)
    canvas.paste(_c15, (48, 1670), _c15)
except Exception:
    pass
layout["Bs"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/16_icon_Like.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1284, 1888), _c16)
except Exception:
    pass
layout["Like"] = [1284, 1888, 1428, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/17_icon_4.36.png
try:
    _c17 = get_crop(17, 62, 64)
    canvas.paste(_c17, (179, 1), _c17)
except Exception:
    pass
layout["4.36"] = [179, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 97, 59)
    canvas.paste(_c18, (1216, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [1216, 1, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/19_icon_Like.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1284, 1405), _c19)
except Exception:
    pass
layout["Like"] = [1284, 1405, 1428, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 58, 62)
    canvas.paste(_c20, (312, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [312, 2, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 54, 59)
    canvas.paste(_c21, (1318, 1), _c21)
except Exception:
    pass
layout["icon_21"] = [1318, 1, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/22_icon_Constellations_Counselling.png
try:
    _c22 = get_crop(22, 1344, 232)
    canvas.paste(_c22, (48, 2092), _c22)
except Exception:
    pass
layout["Constellations_Counsellin"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/23_icon_4.36.png
try:
    _c23 = get_crop(23, 58, 65)
    canvas.paste(_c23, (116, 1), _c23)
except Exception:
    pass
layout["4.36"] = [116, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 53, 62)
    canvas.paste(_c24, (247, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [247, 2, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/25_icon_Share.png
try:
    _c25 = get_crop(25, 120, 144)
    canvas.paste(_c25, (1164, 1888), _c25)
except Exception:
    pass
layout["Share"] = [1164, 1888, 1284, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/26_icon_Share.png
try:
    _c26 = get_crop(26, 120, 144)
    canvas.paste(_c26, (1164, 1405), _c26)
except Exception:
    pass
layout["Share"] = [1164, 1405, 1284, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 62)
    canvas.paste(_c27, (383, 2), _c27)
except Exception:
    pass
layout["icon_27"] = [383, 2, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/28_icon_Oakland_Ballers_Meet_Greet_at_Sports.png
try:
    _c28 = get_crop(28, 1344, 326)
    canvas.paste(_c28, (48, 1670), _c28)
except Exception:
    pass
layout["Oakland_Ballers_Meet_&_Gr"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/29_icon_Free.png
try:
    _c29 = get_crop(29, 139, 124)
    canvas.paste(_c29, (97, 2566), _c29)
except Exception:
    pass
layout["Free"] = [97, 2566, 236, 2690]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/30_icon_Free.png
try:
    _c30 = get_crop(30, 75, 72)
    canvas.paste(_c30, (249, 2588), _c30)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/31_icon_Art_for_Grief_and_Loss.png
try:
    _c31 = get_crop(31, 1344, 232)
    canvas.paste(_c31, (48, 2092), _c31)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/32_icon_sGrow.png
try:
    _c32 = get_crop(32, 128, 62)
    canvas.paste(_c32, (401, 1380), _c32)
except Exception:
    pass
layout["sGrow"] = [401, 1380, 529, 1442]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/33_icon_Report_event.png
try:
    _c33 = get_crop(33, 246, 144)
    canvas.paste(_c33, (829, 766), _c33)
except Exception:
    pass
layout["Report_event"] = [829, 766, 1075, 910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/34_icon_Realtors_Refine_Client_Appreciation.png
try:
    _c34 = get_crop(34, 1344, 387)
    canvas.paste(_c34, (48, 1187), _c34)
except Exception:
    pass
layout["Realtors:_Refine_Client_A"] = [48, 1187, 1392, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/35_icon_icon_35.png
try:
    _c35 = get_crop(35, 71, 81)
    canvas.paste(_c35, (1320, 2275), _c35)
except Exception:
    pass
layout["icon_35"] = [1320, 2275, 1391, 2356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/36_icon_Backpacking_Clinic_W.png
try:
    _c36 = get_crop(36, 307, 144)
    canvas.paste(_c36, (286, 317), _c36)
except Exception:
    pass
layout["Backpacking_Clinic_W__"] = [286, 317, 593, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/37_icon_Wednesday_May_8_I_00_PM.png
try:
    _c37 = get_crop(37, 1344, 232)
    canvas.paste(_c37, (48, 2092), _c37)
except Exception:
    pass
layout["Wednesday;_May_8,_I:00_PM"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/38_icon_Sports_Basement.png
try:
    _c38 = get_crop(38, 1344, 326)
    canvas.paste(_c38, (48, 1670), _c38)
except Exception:
    pass
layout["Sports_Basement"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/39_icon_Contact_the.png
try:
    _c39 = get_crop(39, 416, 144)
    canvas.paste(_c39, (365, 766), _c39)
except Exception:
    pass
layout["Contact_the"] = [365, 766, 781, 910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/40_text_4.36.png
try:
    _c40 = get_crop(40, 89, 45)
    canvas.paste(_c40, (22, 15), _c40)
except Exception:
    pass
layout["4.36"] = [22, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/41_text_More_like_this.png
try:
    _c41 = get_crop(41, 367, 61)
    canvas.paste(_c41, (45, 1072), _c41)
except Exception:
    pass
layout["More_like_this"] = [45, 1072, 412, 1133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_10_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-12/42_text_General_Admission.png
try:
    _c42 = get_crop(42, 75, 72)
    canvas.paste(_c42, (249, 2588), _c42)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
