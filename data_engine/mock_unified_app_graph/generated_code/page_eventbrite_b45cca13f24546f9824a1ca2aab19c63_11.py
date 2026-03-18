# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_11
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13.png
# step_index: 11/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the Eventbrite page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Canvas base (ensure white background)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar (top area ~56px) - subtle desaturated green/grey like the screenshot
status_height = 56
draw.rectangle([(0, 0), (1440, status_height)], fill="#9aa69f")

# Notification banner below status bar (pale green)
notif_top = status_height
notif_bottom = 140
draw.rectangle([(0, notif_top), (1440, notif_bottom)], fill="#e8f6ee")
# thin divider under notification
draw.line([(24, notif_bottom), (1416, notif_bottom)], fill="#d3e8db", width=1)

# Hero image area (dark band behind the event image)
hero_top = notif_bottom
hero_bottom = 420
draw.rectangle([(0, hero_top), (1440, hero_bottom)], fill="#1f1f1f")
# subtle dark overlay bar near bottom of hero to mimic progress/overlay area
overlay_h = 12
overlay_y = hero_bottom - 44
# center thin rounded-like progress segments (as subtle strokes, not icons/text)
seg_w = 160
gap = 28
start_x = 160
for i in range(7):
    x0 = start_x + i * (seg_w + gap)
    x1 = x0 + seg_w
    draw.rectangle([(x0, overlay_y), (x1, overlay_y + overlay_h)], fill="#333333")

# Main content background remains white; draw a faint top padding divider
draw.line([(24, hero_bottom + 18), (1416, hero_bottom + 18)], fill="#f2f2f2", width=1)

# Organizer card (rounded rect behind organizer avatar/name/follow button)
org_card_x0 = 40
org_card_x1 = 1400
org_card_y0 = 960
org_card_y1 = 1128
org_radius = 28
draw.rounded_rectangle([(org_card_x0, org_card_y0), (org_card_x1, org_card_y1)],
                       radius=org_radius, fill="#f7f6fb", outline="#e9e6f2", width=2)

# Small divider line under the organizer/refund area
divider_y = 1520
draw.line([(40, divider_y), (1400, divider_y)], fill="#f0f0f1", width=2)

# Section separator near where date selection begins
date_section_top = 1720
draw.rectangle([(0, date_section_top), (1440, date_section_top+8)], fill="#ffffff")
# Light shadow line above date cards area
draw.line([(24, date_section_top+8), (1416, date_section_top+8)], fill="#efeff2", width=1)

# Date cards container background (subtle off-white strip behind the horizontally scrollable date cards)
date_container_y0 = 1760
date_container_y1 = 2120
draw.rectangle([(24, date_container_y0), (1416, date_container_y1)], fill="#ffffff")

# Add faint card separators for the date cards row (visual structure only)
card_w = 420
card_h = 320
card_gap = 36
cards_x = 40
for i in range(4):
    cx0 = cards_x + i * (card_w + card_gap)
    cx1 = cx0 + card_w
    cy0 = date_container_y0 + 16
    cy1 = cy0 + card_h
    # Unselected card background (very light)
    draw.rounded_rectangle([(cx0, cy0), (cx1, cy1)], radius=20, fill="#ffffff", outline="#efeef6", width=2)

# Selected date card accent (leftmost) - blue accent border only (no text/icons drawn)
sel_x0 = 40
sel_y0 = date_container_y0 + 16
sel_x1 = sel_x0 + card_w
sel_y1 = sel_y0 + card_h
draw.rounded_rectangle([(sel_x0, sel_y0), (sel_x1, sel_y1)], radius=20, fill=None, outline="#2f58ff", width=6)

# Thin divider below date cards
draw.line([(24, date_container_y1 + 8), (1416, date_container_y1 + 8)], fill="#efeef2", width=1)

# Ticket selection card area above checkout button
ticket_card_top = 2160
ticket_card_bottom = 2320  # keep above the checkout area that will be pasted on top
ticket_x0 = 40
ticket_x1 = 1400
ticket_radius = 20
# White card with prominent blue border to match the ticket selector structure
draw.rounded_rectangle([(ticket_x0, ticket_card_top), (ticket_x1, ticket_card_bottom)],
                       radius=ticket_radius, fill="#ffffff", outline="#2f58ff", width=6)

# Inner subtle shadow/panel inside ticket card (to indicate content area without drawing text)
inner_margin = 28
draw.rounded_rectangle([(ticket_x0 + inner_margin, ticket_card_top + inner_margin),
                        (ticket_x1 - inner_margin, ticket_card_bottom - inner_margin)],
                       radius=14, fill="#ffffff", outline="#efeff6", width=1)

# Add a faint horizontal rule separating content sections above the ticket card
draw.line([(40, ticket_card_top - 28), (1400, ticket_card_top - 28)], fill="#f3f3f5", width=1)

# Final top-of-page left padding guideline (subtle vertical rule to echo layout margins)
draw.line([(40, hero_bottom + 6), (40, 2200)], fill="#ffffff00")  # effectively no-op (keeps layout intent)

# Note: Do not draw any icons, text, or buttons — those will be pasted separately at their exact positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/01_icon_April.png
try:
    _c1 = get_crop(1, 450, 352)
    canvas.paste(_c1, (24, 1972), _c1)
except Exception:
    pass
layout["April"] = [24, 1972, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/02_icon_April.png
try:
    _c2 = get_crop(2, 450, 352)
    canvas.paste(_c2, (474, 1972), _c2)
except Exception:
    pass
layout["April"] = [474, 1972, 924, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/03_icon_27.png
try:
    _c3 = get_crop(3, 111, 104)
    canvas.paste(_c3, (988, 2440), _c3)
except Exception:
    pass
layout["27"] = [988, 2440, 1099, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/04_icon_Check_out_for_S35.00.png
try:
    _c4 = get_crop(4, 1440, 636)
    canvas.paste(_c4, (0, 2324), _c4)
except Exception:
    pass
layout["Check_out_for_S35.00"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/05_icon_27.png
try:
    _c5 = get_crop(5, 450, 352)
    canvas.paste(_c5, (924, 1972), _c5)
except Exception:
    pass
layout["27"] = [924, 1972, 1374, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/06_icon_27.png
try:
    _c6 = get_crop(6, 108, 104)
    canvas.paste(_c6, (1215, 2441), _c6)
except Exception:
    pass
layout["27"] = [1215, 2441, 1323, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/07_icon_April.png
try:
    _c7 = get_crop(7, 450, 352)
    canvas.paste(_c7, (924, 1972), _c7)
except Exception:
    pass
layout["April"] = [924, 1972, 1374, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/08_icon_27.png
try:
    _c8 = get_crop(8, 90, 101)
    canvas.paste(_c8, (1109, 2443), _c8)
except Exception:
    pass
layout["27"] = [1109, 2443, 1199, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 67)
    canvas.paste(_c9, (1154, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1154, 1, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/10_icon_Kangalee_Arts_Ensemble_Inc..png
try:
    _c10 = get_crop(10, 629, 144)
    canvas.paste(_c10, (288, 1028), _c10)
except Exception:
    pass
layout["Kangalee_Arts_Ensemble,_I"] = [288, 1028, 917, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 57, 58)
    canvas.paste(_c11, (312, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [312, 5, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/12_icon_Dismiss_notification.png
try:
    _c12 = get_crop(12, 142, 142)
    canvas.paste(_c12, (1251, 97), _c12)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 89, 66)
    canvas.paste(_c13, (1211, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1211, 1, 1300, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/14_icon_7.06.png
try:
    _c14 = get_crop(14, 57, 60)
    canvas.paste(_c14, (182, 3), _c14)
except Exception:
    pass
layout["7.06"] = [182, 3, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 46, 53)
    canvas.paste(_c15, (252, 8), _c15)
except Exception:
    pass
layout["icon_15"] = [252, 8, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/16_icon_7.06.png
try:
    _c16 = get_crop(16, 59, 63)
    canvas.paste(_c16, (115, 1), _c16)
except Exception:
    pass
layout["7.06"] = [115, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 65)
    canvas.paste(_c17, (1319, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 1, 1371, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/18_icon_26.png
try:
    _c18 = get_crop(18, 450, 352)
    canvas.paste(_c18, (474, 1972), _c18)
except Exception:
    pass
layout["26"] = [474, 1972, 924, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 52, 61)
    canvas.paste(_c19, (382, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [382, 3, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/20_icon_Thursday_April_25.png
try:
    _c20 = get_crop(20, 181, 55)
    canvas.paste(_c20, (252, 541), _c20)
except Exception:
    pass
layout["Thursday_April_25"] = [252, 541, 433, 596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/21_icon_S35.00.png
try:
    _c21 = get_crop(21, 100, 103)
    canvas.paste(_c21, (291, 2576), _c21)
except Exception:
    pass
layout["S35.00"] = [291, 2576, 391, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/22_icon_2_hrs_30_mins.png
try:
    _c22 = get_crop(22, 296, 71)
    canvas.paste(_c22, (133, 1442), _c22)
except Exception:
    pass
layout["2_hrs_30_mins"] = [133, 1442, 429, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 42, 55)
    canvas.paste(_c23, (1272, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1272, 7, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/24_icon_25.png
try:
    _c24 = get_crop(24, 450, 352)
    canvas.paste(_c24, (24, 1972), _c24)
except Exception:
    pass
layout["25"] = [24, 1972, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/25_text_7.06.png
try:
    _c25 = get_crop(25, 93, 49)
    canvas.paste(_c25, (19, 12), _c25)
except Exception:
    pass
layout["7.06"] = [19, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/26_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (36, 108), _c26)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/27_text_Thursday_April_25.png
try:
    _c27 = get_crop(27, 456, 77)
    canvas.paste(_c27, (40, 758), _c27)
except Exception:
    pass
layout["Thursday_April_25"] = [40, 758, 496, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/28_text_8_00_PM.png
try:
    _c28 = get_crop(28, 215, 63)
    canvas.paste(_c28, (520, 762), _c28)
except Exception:
    pass
layout["8:00_PM"] = [520, 762, 735, 825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/29_text_THE_LIFE_DEATH_OF_ART.png
try:
    _c29 = get_crop(29, 629, 144)
    canvas.paste(_c29, (288, 1028), _c29)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [288, 1028, 917, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/30_text_JACK.png
try:
    _c30 = get_crop(30, 121, 52)
    canvas.paste(_c30, (137, 1341), _c30)
except Exception:
    pass
layout["JACK"] = [137, 1341, 258, 1393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/31_text_Refund_policy.png
try:
    _c31 = get_crop(31, 299, 63)
    canvas.paste(_c31, (138, 1558), _c31)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/32_text_The_organizer_will_review_refund_request.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1295), _c32)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/33_text_Select_date_and_time.png
try:
    _c33 = get_crop(33, 450, 352)
    canvas.paste(_c33, (24, 1972), _c33)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 1972, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/34_text_General_Admission.png
try:
    _c34 = get_crop(34, 415, 55)
    canvas.paste(_c34, (116, 2451), _c34)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/35_text_S35.00.png
try:
    _c35 = get_crop(35, 163, 57)
    canvas.paste(_c35, (113, 2592), _c35)
except Exception:
    pass
layout["S35.00"] = [113, 2592, 276, 2649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_11_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-13/36_clickable_Organizer_profile_picture.png
try:
    _c36 = get_crop(36, 144, 144)
    canvas.paste(_c36, (96, 1067), _c36)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
