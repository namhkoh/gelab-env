# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_10
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12.png
# step_index: 10/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the Event page UI
# Uses provided variables: canvas (1440x2960 RGB) and draw (PIL ImageDraw)
# font_sm, font_md, font_lg, font_xl are available but not used (structure only)

# Overall background (slight off-white to match screenshot)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 249, 252))

# Status bar (top area, dark/neutral to host time/signal icons)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(64, 64, 64))

# Top toolbar / hero image area (dark image/banner background)
hero_top = status_h
hero_bottom = 520
draw.rectangle((0, hero_top, 1440, hero_bottom), fill=(26, 26, 26))

# Subtle darker overlay band at bottom of hero to suggest fade/overlay
overlay_h = 48
for i in range(overlay_h):
    alpha = int(6 + (i * 4))  # simulate a gradient by slight color steps
    step_color = (26 + alpha, 26 + alpha, 26 + alpha)
    draw.rectangle((0, hero_bottom - overlay_h + i, 1440, hero_bottom - overlay_h + i + 1), fill=step_color)

# Progress indicator bar (part of hero background)
pb_w = 1100
pb_h = 10
pb_x = (1440 - pb_w) // 2
pb_y = hero_bottom - 38
# full track
draw.rounded_rectangle((pb_x, pb_y, pb_x + pb_w, pb_y + pb_h), radius=6, fill=(60, 60, 60))
# a few segments to mimic progress markers (light)
seg_w = pb_w // 8
seg_h = pb_h - 4
seg_y = pb_y + 2
for i in range(8):
    seg_x = pb_x + 8 + i * seg_w
    draw.rectangle((seg_x, seg_y, seg_x + seg_w - 14, seg_y + seg_h), fill=(170, 170, 170))

# Thin divider under hero
draw.line((24, hero_bottom + 8, 1440 - 24, hero_bottom + 8), fill=(230, 228, 235), width=2)

# Organizer card background (rounded rectangle behind organizer avatar/name/follow)
org_x = 24
org_y = 980
org_w = 1440 - 48
org_h = 176
org_radius = 24
draw.rounded_rectangle((org_x, org_y, org_x + org_w, org_y + org_h), radius=org_radius, fill=(245, 244, 247))
# subtle top highlight
draw.line((org_x + 6, org_y + 6, org_x + org_w - 6, org_y + 6), fill=(250, 250, 251), width=1)
# subtle bottom divider for the card
draw.line((org_x + 12, org_y + org_h - 8, org_x + org_w - 12, org_y + org_h - 8), fill=(233, 231, 238), width=1)

# Thin separator line below organizer/details area
sep_y = org_y + org_h + 70
draw.line((24, sep_y, 1440 - 24, sep_y), fill=(239, 238, 242), width=2)

# Small info separators (light horizontal lines to structure the details)
info_start = sep_y + 24
for j in range(2):
    y = info_start + j * 84
    draw.line((48, y, 1440 - 48, y), fill=(246, 245, 248), width=1)

# "Select date and time" cards container area - subtle background block (keeps page airy)
cards_top = 1880
cards_bottom = 2760
draw.rectangle((0, cards_top - 80, 1440, cards_top + 660), fill=(250, 249, 252))

# Date/time cards (3 cards horizontally) - draw only card backgrounds and borders (no content)
card_w = 450
card_h = 516
card_radius = 20

# Positions provided by detection: (24,1972), (474,1972), (924,1972)
card_positions = [(24, 1972), (474, 1972), (924, 1972)]
# Standard card fill and border colors
card_fill = (255, 255, 255)
card_border = (233, 230, 241)
selected_border = (47, 82, 255)  # blue accent for selected card

for idx, (cx, cy) in enumerate(card_positions):
    x1, y1 = cx, cy
    x2, y2 = cx + card_w, cy + card_h
    # Draw card background
    draw.rounded_rectangle((x1, y1, x2, y2), radius=card_radius, fill=card_fill, outline=card_border, width=4)
    # For the first card, draw the selected blue border thicker
    if idx == 0:
        draw.rounded_rectangle((x1 + 6, y1 + 6, x2 - 6, y2 - 6), radius=card_radius - 6, outline=selected_border, width=8)

# Separator under date/time cards
cards_sep_y = card_positions[0][1] + card_h + 40
draw.line((24, cards_sep_y, 1440 - 24, cards_sep_y), fill=(236, 235, 240), width=2)

# "About this event" pill/tag background (rounded capsule behind category label)
pill_x = 48
pill_y = 2688
pill_w = 720
pill_h = 84
pill_radius = pill_h // 2
draw.rounded_rectangle((pill_x, pill_y, pill_x + pill_w, pill_y + pill_h), radius=pill_radius, fill=(244, 245, 249))

# Bottom page divider line
draw.line((24, 2920, 1440 - 24, 2920), fill=(245, 244, 246), width=1)

# Final subtle horizontal guides for sections (light greys)
section_guides = [hero_bottom + 40, org_y + org_h + 40, cards_sep_y + 80]
for gy in section_guides:
    draw.line((48, gy, 1440 - 48, gy), fill=(250, 249, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/01_icon_25.png
try:
    _c1 = get_crop(1, 450, 516)
    canvas.paste(_c1, (24, 1972), _c1)
except Exception:
    pass
layout["25"] = [24, 1972, 474, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/02_icon_26.png
try:
    _c2 = get_crop(2, 450, 516)
    canvas.paste(_c2, (474, 1972), _c2)
except Exception:
    pass
layout["26"] = [474, 1972, 924, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/03_icon_7.06.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["7.06"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/04_icon_27.png
try:
    _c4 = get_crop(4, 450, 516)
    canvas.paste(_c4, (924, 1972), _c4)
except Exception:
    pass
layout["27"] = [924, 1972, 1374, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/05_icon_Cha.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1260, 108), _c5)
except Exception:
    pass
layout["Cha"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 44, 66)
    canvas.paste(_c6, (1158, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1158, 2, 1202, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/07_icon_Arts.png
try:
    _c7 = get_crop(7, 724, 98)
    canvas.paste(_c7, (36, 2735), _c7)
except Exception:
    pass
layout["Arts"] = [36, 2735, 760, 2833]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/08_icon_Cha.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1116, 108), _c8)
except Exception:
    pass
layout["Cha"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 46, 61)
    canvas.paste(_c9, (1326, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [1326, 4, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/10_icon_Kangalee_Arts_Ensemble_Inc..png
try:
    _c10 = get_crop(10, 629, 144)
    canvas.paste(_c10, (288, 1028), _c10)
except Exception:
    pass
layout["Kangalee_Arts_Ensemble,_I"] = [288, 1028, 917, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/11_icon_7.06.png
try:
    _c11 = get_crop(11, 61, 65)
    canvas.paste(_c11, (181, 1), _c11)
except Exception:
    pass
layout["7.06"] = [181, 1, 242, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 55, 65)
    canvas.paste(_c12, (247, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 1, 302, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/13_icon_7.06.png
try:
    _c13 = get_crop(13, 59, 66)
    canvas.paste(_c13, (116, 0), _c13)
except Exception:
    pass
layout["7.06"] = [116, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 64, 66)
    canvas.paste(_c14, (309, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [309, 1, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/15_icon_Thursday_April_25.png
try:
    _c15 = get_crop(15, 181, 60)
    canvas.paste(_c15, (252, 536), _c15)
except Exception:
    pass
layout["Thursday_April_25"] = [252, 536, 433, 596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 68)
    canvas.paste(_c16, (382, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 1, 432, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 97, 61)
    canvas.paste(_c17, (1217, 4), _c17)
except Exception:
    pass
layout["icon_17"] = [1217, 4, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/18_text_7.06.png
try:
    _c18 = get_crop(18, 92, 41)
    canvas.paste(_c18, (22, 17), _c18)
except Exception:
    pass
layout["7.06"] = [22, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/19_text_Cha.png
try:
    _c19 = get_crop(19, 37, 18)
    canvas.paste(_c19, (1245, 288), _c19)
except Exception:
    pass
layout["Cha"] = [1245, 288, 1282, 306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/20_text_Thursday_April_25.png
try:
    _c20 = get_crop(20, 456, 77)
    canvas.paste(_c20, (40, 758), _c20)
except Exception:
    pass
layout["Thursday_April_25"] = [40, 758, 496, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/21_text_8_00_PM.png
try:
    _c21 = get_crop(21, 215, 63)
    canvas.paste(_c21, (520, 762), _c21)
except Exception:
    pass
layout["8:00_PM"] = [520, 762, 735, 825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/22_text_THE_LIFE_DEATH_OF_ART.png
try:
    _c22 = get_crop(22, 629, 144)
    canvas.paste(_c22, (288, 1028), _c22)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [288, 1028, 917, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/23_text_JACK.png
try:
    _c23 = get_crop(23, 121, 52)
    canvas.paste(_c23, (137, 1341), _c23)
except Exception:
    pass
layout["JACK"] = [137, 1341, 258, 1393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/24_text_2_hrs_30_mins.png
try:
    _c24 = get_crop(24, 290, 54)
    canvas.paste(_c24, (141, 1450), _c24)
except Exception:
    pass
layout["2_hrs_30_mins"] = [141, 1450, 431, 1504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1558), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1295), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/27_text_Select_date_and_time.png
try:
    _c27 = get_crop(27, 450, 516)
    canvas.paste(_c27, (24, 1972), _c27)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 1972, 474, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/28_text_Saturday.png
try:
    _c28 = get_crop(28, 190, 62)
    canvas.paste(_c28, (1053, 2043), _c28)
except Exception:
    pass
layout["Saturday"] = [1053, 2043, 1243, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/29_text_April.png
try:
    _c29 = get_crop(29, 450, 516)
    canvas.paste(_c29, (924, 1972), _c29)
except Exception:
    pass
layout["April"] = [924, 1972, 1374, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/30_text_About_this_event.png
try:
    _c30 = get_crop(30, 452, 65)
    canvas.paste(_c30, (45, 2645), _c30)
except Exception:
    pass
layout["About_this_event"] = [45, 2645, 497, 2710]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/31_text_Anew_drama_about_art_genocide_capitalism.png
try:
    _c31 = get_crop(31, 450, 516)
    canvas.paste(_c31, (474, 1972), _c31)
except Exception:
    pass
layout["Anew_drama_about_art,_gen"] = [474, 1972, 924, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_10_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-12/32_clickable_Organizer_profile_picture.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (96, 1067), _c32)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
