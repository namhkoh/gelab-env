# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_05
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7.png
# step_index: 5/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural UI backgrounds on provided canvas using the provided draw object.
w, h = canvas.size

# Colors
status_bg = (236, 236, 236)      # light grey status bar
status_div = (210, 210, 210)     # divider under status
hero_start = (38, 30, 45)        # dark purple-ish for hero top
hero_end = (230, 230, 235)       # light for hero bottom gradient
hero_overlay_dark = (20, 14, 22) # darker overlay near bottom of hero
card_shadow = (233, 233, 233)    # card shadow
card_bg = (247, 248, 250)        # card background (very light)
section_div = (235, 235, 238)    # light section divider
pill_bg = (241, 245, 250)        # tag/pill background
agenda_bg = (255, 244, 242)      # pale peach for agenda item
accent_peach = (255, 182, 162)   # accent strip for agenda

# 1) Status bar area (~50px)
status_h = 50
draw.rectangle([0, 0, w, status_h], fill=status_bg)
draw.line([(0, status_h), (w, status_h)], fill=status_div, width=1)

# 2) Hero/banner image background with vertical gradient (under top icons)
hero_y0 = status_h
hero_y1 = status_h + 320
for y in range(hero_y0, hero_y1):
    t = (y - hero_y0) / max(1, (hero_y1 - hero_y0 - 1))
    r = int(hero_start[0] * (1 - t) + hero_end[0] * t)
    g = int(hero_start[1] * (1 - t) + hero_end[1] * t)
    b = int(hero_start[2] * (1 - t) + hero_end[2] * t)
    draw.line([(0, y), (w, y)], fill=(r, g, b))

# subtle dark overlay at bottom of hero for contrast where title sits
overlay_h = 72
for i in range(overlay_h):
    y = hero_y1 - overlay_h + i
    t = i / max(1, overlay_h - 1)
    # blend between transparent (use hero_end) and hero_overlay_dark
    r = int(hero_end[0] * (1 - t) + hero_overlay_dark[0] * t)
    g = int(hero_end[1] * (1 - t) + hero_overlay_dark[1] * t)
    b = int(hero_end[2] * (1 - t) + hero_overlay_dark[2] * t)
    draw.line([(0, y), (w, y)], fill=(r, g, b))

# 3) Main content background (canvas default white) - leave as is

# 4) Organizer/info card (rounded rectangle with subtle shadow)
card_margin = 48
card_left = card_margin
card_right = w - card_margin
card_y1 = 1080
card_y2 = 1220
card_radius = 28
# shadow
draw.rounded_rectangle(
    [card_left, card_y1 + 6, card_right, card_y2 + 6],
    radius=card_radius,
    fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [card_left, card_y1, card_right, card_y2],
    radius=card_radius,
    fill=card_bg
)

# 5) Thin divider under the organizer/info area
div_y = card_y2 + 60
draw.line([(card_left, div_y), (card_right, div_y)], fill=section_div, width=1)

# 6) Small separator line under top content (near where event details list ends)
sep_y1 = div_y + 220
draw.line([(card_left, sep_y1), (card_right, sep_y1)], fill=section_div, width=1)

# 7) "About this event" tag pill background (rounded)
pill_x = card_left
pill_w = 420
pill_h = 64
pill_y = 2030  # positioned below "About this event" header; icons/text will be pasted on top
draw.rounded_rectangle(
    [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
    radius=pill_h // 2,
    fill=pill_bg
)

# 8) Another subtle divider a bit lower (before agenda)
lower_div_y = 2460
draw.line([(card_left, lower_div_y), (card_right, lower_div_y)], fill=section_div, width=1)

# 9) Agenda item background card (large rounded rectangle with pale peach background)
agenda_x1 = card_left
agenda_x2 = card_right
agenda_y1 = 2580
agenda_y2 = 2900
agenda_radius = 22
# shadow for agenda
draw.rounded_rectangle(
    [agenda_x1, agenda_y1 + 6, agenda_x2, agenda_y2 + 6],
    radius=agenda_radius,
    fill=card_shadow
)
draw.rounded_rectangle(
    [agenda_x1, agenda_y1, agenda_x2, agenda_y2],
    radius=agenda_radius,
    fill=agenda_bg
)
# vertical accent bar on left of agenda item
accent_x = agenda_x1 + 20
accent_w = 8
accent_y_top = agenda_y1 + 28
accent_y_bot = agenda_y2 - 28
draw.rectangle([accent_x, accent_y_top, accent_x + accent_w, accent_y_bot], fill=accent_peach)

# 10) Horizontal content separator near bottom of page
bottom_sep_y = 2940
draw.line([(card_left, bottom_sep_y), (card_right, bottom_sep_y)], fill=section_div, width=1)

# No text/icons drawn; all content elements (texts, avatars, buttons, icons) will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1163), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/01_icon_Jon_Holmes.png
try:
    _c1 = get_crop(1, 240, 264)
    canvas.paste(_c1, (956, 374), _c1)
except Exception:
    pass
layout["Jon_Holmes"] = [956, 374, 1196, 638]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/02_icon_5.20.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["5.20"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/03_icon_Registration_Below.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Registration_Below"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/04_icon_Business_Professional.png
try:
    _c4 = get_crop(4, 234, 144)
    canvas.paste(_c4, (48, 2300), _c4)
except Exception:
    pass
layout["Business_&_Professional"] = [48, 2300, 282, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/05_icon_Registration_Below.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["Registration_Below"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/06_icon_SEMINAR.png
try:
    _c6 = get_crop(6, 62, 69)
    canvas.paste(_c6, (179, 1), _c6)
except Exception:
    pass
layout["SEMINAR"] = [179, 1, 241, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/07_icon_Registration_Below.png
try:
    _c7 = get_crop(7, 43, 65)
    canvas.paste(_c7, (1272, 1), _c7)
except Exception:
    pass
layout["Registration_Below"] = [1272, 1, 1315, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/08_icon_TEXAS_PROPERTY_PROTEST.png
try:
    _c8 = get_crop(8, 70, 67)
    canvas.paste(_c8, (307, 1), _c8)
except Exception:
    pass
layout["TEXAS_PROPERTY_PROTEST"] = [307, 1, 377, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 67)
    canvas.paste(_c9, (1317, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1317, 1, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/10_icon_5.20.png
try:
    _c10 = get_crop(10, 61, 69)
    canvas.paste(_c10, (115, 0), _c10)
except Exception:
    pass
layout["5.20"] = [115, 0, 176, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/11_icon_Registration_Below.png
try:
    _c11 = get_crop(11, 58, 64)
    canvas.paste(_c11, (1213, 1), _c11)
except Exception:
    pass
layout["Registration_Below"] = [1213, 1, 1271, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/12_icon_SEMINAR.png
try:
    _c12 = get_crop(12, 54, 68)
    canvas.paste(_c12, (247, 1), _c12)
except Exception:
    pass
layout["SEMINAR"] = [247, 1, 301, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/13_icon_Quinton_Starks.png
try:
    _c13 = get_crop(13, 254, 274)
    canvas.paste(_c13, (679, 368), _c13)
except Exception:
    pass
layout["Quinton_Starks"] = [679, 368, 933, 642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/14_icon_TEXAS_PROPERTY_PROTEST.png
try:
    _c14 = get_crop(14, 47, 67)
    canvas.paste(_c14, (384, 1), _c14)
except Exception:
    pass
layout["TEXAS_PROPERTY_PROTEST"] = [384, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/15_icon_TEXAS_PROPERTY_PROTEST.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1116, 108), _c15)
except Exception:
    pass
layout["TEXAS_PROPERTY_PROTEST"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/16_icon_Going_over_the_four_step_process_for.png
try:
    _c16 = get_crop(16, 1440, 259)
    canvas.paste(_c16, (0, 2701), _c16)
except Exception:
    pass
layout["Going_over_the_four_step_"] = [0, 2701, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/17_icon_Read_more.png
try:
    _c17 = get_crop(17, 234, 144)
    canvas.paste(_c17, (48, 2300), _c17)
except Exception:
    pass
layout["Read_more"] = [48, 2300, 282, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/18_text_5.20.png
try:
    _c18 = get_crop(18, 89, 43)
    canvas.paste(_c18, (22, 17), _c18)
except Exception:
    pass
layout["5.20"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/19_text_starks.png
try:
    _c19 = get_crop(19, 230, 81)
    canvas.paste(_c19, (340, 387), _c19)
except Exception:
    pass
layout["starks"] = [340, 387, 570, 468]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/20_text_IITH.png
try:
    _c20 = get_crop(20, 57, 27)
    canvas.paste(_c20, (275, 541), _c20)
except Exception:
    pass
layout["IITH"] = [275, 541, 332, 568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/21_text_IOam.png
try:
    _c21 = get_crop(21, 62, 25)
    canvas.paste(_c21, (363, 541), _c21)
except Exception:
    pass
layout["IOam"] = [363, 541, 425, 566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/22_text_i30am.png
try:
    _c22 = get_crop(22, 80, 25)
    canvas.paste(_c22, (435, 541), _c22)
except Exception:
    pass
layout["i30am"] = [435, 541, 515, 566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/23_text_via_the_Zoom_App.png
try:
    _c23 = get_crop(23, 178, 29)
    canvas.paste(_c23, (365, 569), _c23)
except Exception:
    pass
layout["via_the_Zoom_App"] = [365, 569, 543, 598]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/24_text_Saturday.png
try:
    _c24 = get_crop(24, 252, 77)
    canvas.paste(_c24, (38, 758), _c24)
except Exception:
    pass
layout["Saturday;"] = [38, 758, 290, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/25_text_11.png
try:
    _c25 = get_crop(25, 64, 50)
    canvas.paste(_c25, (407, 770), _c25)
except Exception:
    pass
layout["11"] = [407, 770, 471, 820]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/26_text_IO_O0_AM.png
try:
    _c26 = get_crop(26, 244, 62)
    canvas.paste(_c26, (512, 763), _c26)
except Exception:
    pass
layout["IO:O0_AM"] = [512, 763, 756, 825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/27_text_ZOOM_Texas_Property_Tax_Protest.png
try:
    _c27 = get_crop(27, 318, 144)
    canvas.paste(_c27, (288, 1123), _c27)
except Exception:
    pass
layout["ZOOM_Texas_Property_Tax_P"] = [288, 1123, 606, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/28_text_Seminar.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (96, 1162), _c28)
except Exception:
    pass
layout["Seminar"] = [96, 1162, 240, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/29_text_Quinton_Starks.png
try:
    _c29 = get_crop(29, 318, 144)
    canvas.paste(_c29, (288, 1123), _c29)
except Exception:
    pass
layout["Quinton_Starks"] = [288, 1123, 606, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/30_text_16_Followers.png
try:
    _c30 = get_crop(30, 318, 144)
    canvas.paste(_c30, (288, 1123), _c30)
except Exception:
    pass
layout["16_Followers"] = [288, 1123, 606, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/31_text_Online_event.png
try:
    _c31 = get_crop(31, 274, 54)
    canvas.paste(_c31, (139, 1436), _c31)
except Exception:
    pass
layout["Online_event"] = [139, 1436, 413, 1490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/32_text_1_hrs_30_mins.png
try:
    _c32 = get_crop(32, 285, 52)
    canvas.paste(_c32, (146, 1545), _c32)
except Exception:
    pass
layout["1_hrs_30_mins"] = [146, 1545, 431, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/33_text_Refund_policy.png
try:
    _c33 = get_crop(33, 299, 63)
    canvas.paste(_c33, (138, 1653), _c33)
except Exception:
    pass
layout["Refund_policy"] = [138, 1653, 437, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/34_text_The_organizer_will_review_refund_request.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 1390), _c34)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/35_text_About_this_event.png
try:
    _c35 = get_crop(35, 452, 61)
    canvas.paste(_c35, (45, 1953), _c35)
except Exception:
    pass
layout["About_this_event"] = [45, 1953, 497, 2014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/36_text_Learn_how_to_protest_your_property_tax_a.png
try:
    _c36 = get_crop(36, 234, 144)
    canvas.paste(_c36, (48, 2300), _c36)
except Exception:
    pass
layout["Learn_how_to_protest_your"] = [48, 2300, 282, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/37_text_bill._Understand.png
try:
    _c37 = get_crop(37, 300, 45)
    canvas.paste(_c37, (1028, 2189), _c37)
except Exception:
    pass
layout["bill._Understand"] = [1028, 2189, 1328, 2234]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_05_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-7/38_text_Agenda.png
try:
    _c38 = get_crop(38, 227, 74)
    canvas.paste(_c38, (42, 2563), _c38)
except Exception:
    pass
layout["Agenda"] = [42, 2563, 269, 2637]
