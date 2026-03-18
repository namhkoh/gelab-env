# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_10
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12.png
# step_index: 10/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 50)], fill="#cfcfcf")

# Header / hero background (green banner with simple decorative shapes)
hero_top = 50
hero_bottom = 520
draw.rectangle([(0, hero_top), (1440, hero_bottom)], fill="#4a8d62")

# soft darker band at bottom of hero for visual separation
draw.rectangle([(0, hero_bottom-20), (1440, hero_bottom)], fill="#407a56")

# decorative lighter cloud/spot shapes in hero (kept simple, behind icons)
draw.ellipse([(-120, hero_top+40), (420, hero_top+260)], fill="#6fb78a")
draw.ellipse([(980, hero_top+30), (1480, hero_top+210)], fill="#6fb78a")
draw.ellipse([(420, hero_top+10), (1020, hero_top+210)], fill="#3f7f57")

# simple illustrative building blocks (abstract shapes) to suggest the event art
building_x = 160
for w, h, color, offset in [(160, 210, "#2f6fb0", 0), (120, 180, "#3b86c6", 40), (200, 240, "#2b6aa3", -20), (140, 200, "#4a9a6e", 80)]:
    bx = building_x + offset
    draw.rectangle([(bx, hero_top+180), (bx + w, hero_top+180 + h)], fill=color, outline=None)
    # small window rows
    for i in range(3):
        for j in range(2):
            wx = bx + 12 + j * (w//2)
            wy = hero_top+190 + i * 60
            draw.rectangle([(wx, wy), (wx + 28, wy + 36)], fill="#b8e0d0")

# slight vignette/gloss at top center
draw.rectangle([(0, hero_top), (1440, hero_top+12)], fill="#5aa978")

# subtle shadow under the hero image
shadow_top = hero_bottom
for i, a in enumerate([180, 140, 100, 60, 30]):
    alpha = int(a)
    y = shadow_top + i
    draw.line([(0, y), (1440, y)], fill=(220, 220, 220, alpha))

# Main content background remains white (canvas already white) - draw a faint large content card area
content_left = 40
content_right = 1400
# large light content container behind main info section
draw.rounded_rectangle([(content_left, 560), (content_right, 1160)], radius=20, fill="#ffffff", outline=None)

# Organizer small card (rounded rectangle behind organizer row)
org_card_top = 1200
org_card_bottom = 1360
draw.rounded_rectangle([(40, org_card_top), (1400, org_card_bottom)], radius=24, fill="#f4f6f8", outline="#e9ecef", width=2)

# subtle divider under organizer/card area
draw.line([(40, org_card_bottom + 30), (1400, org_card_bottom + 30)], fill="#eceff2", width=1)

# informational icon rows area (left icons + labels area) - just separators and spacing lines
info_start_y = org_card_bottom + 80
draw.line([(40, info_start_y + 160), (1400, info_start_y + 160)], fill="#f0f2f5", width=1)

# About section divider
about_div_y = 2040
draw.line([(40, about_div_y), (1400, about_div_y)], fill="#eceff2", width=2)

# "About this event" area background hint (kept subtle)
draw.rectangle([(40, about_div_y + 20), (1400, about_div_y + 120)], fill="#ffffff", outline=None)

# Ticket selection card (rounded with blue border) - positioned safely above the reserve button area
ticket_card_top = 2140
ticket_card_bottom = 2300  # kept above reserve area (reserve begins at y=2324)
draw.rounded_rectangle([(40, ticket_card_top), (1400, ticket_card_bottom)], radius=18, fill="#ffffff", outline="#294fe4", width=6)

# small inner shadow for ticket card
draw.rectangle([(46, ticket_card_top + 6), (1394, ticket_card_top + 60)], fill="#fafbff", outline=None)

# light separator above bottom area (reserve button region not drawn)
draw.line([(40, 2324), (1400, 2324)], fill="#ece9e6", width=6)

# overall thin content separators at several places
for y in [760, 920, 1520, 1680, 1920]:
    draw.line([(40, y), (1400, y)], fill="#f3f4f6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/01_icon_Reserve_a_spot.png
try:
    _c1 = get_crop(1, 1440, 636)
    canvas.paste(_c1, (0, 2324), _c1)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 111, 105)
    canvas.paste(_c2, (988, 2440), _c2)
except Exception:
    pass
layout["icon_2"] = [988, 2440, 1099, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/03_icon_4.39.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["4.39"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 105, 103)
    canvas.paste(_c4, (1217, 2442), _c4)
except Exception:
    pass
layout["icon_4"] = [1217, 2442, 1322, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 92, 102)
    canvas.paste(_c5, (1108, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/06_icon_Citizens_Sustainability.png
try:
    _c6 = get_crop(6, 773, 144)
    canvas.paste(_c6, (144, 1289), _c6)
except Exception:
    pass
layout["Citizens_Sustainability"] = [144, 1289, 917, 1433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/07_icon_Share.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/08_icon_2PM.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1116, 108), _c8)
except Exception:
    pass
layout["2PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/09_icon_Community_Culture_._City_Town.png
try:
    _c9 = get_crop(9, 1440, 636)
    canvas.paste(_c9, (0, 2324), _c9)
except Exception:
    pass
layout["Community_&_Culture_._Cit"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/10_icon_Ticket_sales_end_soon.png
try:
    _c10 = get_crop(10, 547, 84)
    canvas.paste(_c10, (40, 753), _c10)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 99, 66)
    canvas.paste(_c11, (1215, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1215, 0, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 67)
    canvas.paste(_c12, (1317, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1317, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/13_icon_CLEAN_ENERGY.png
try:
    _c13 = get_crop(13, 232, 61)
    canvas.paste(_c13, (534, 571), _c13)
except Exception:
    pass
layout["CLEAN_ENERGY"] = [534, 571, 766, 632]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/14_icon_PENINSULA.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1116, 108), _c14)
except Exception:
    pass
layout["PENINSULA"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/15_text_4.39.png
try:
    _c15 = get_crop(15, 89, 43)
    canvas.paste(_c15, (22, 17), _c15)
except Exception:
    pass
layout["4.39"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/16_text_my.png
try:
    _c16 = get_crop(16, 45, 39)
    canvas.paste(_c16, (124, 17), _c16)
except Exception:
    pass
layout["my"] = [124, 17, 169, 56]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/17_text_Saturday_April_27.png
try:
    _c17 = get_crop(17, 449, 77)
    canvas.paste(_c17, (38, 885), _c17)
except Exception:
    pass
layout["Saturday,_April_27"] = [38, 885, 487, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/18_text_1I_00AM.png
try:
    _c18 = get_crop(18, 241, 56)
    canvas.paste(_c18, (523, 893), _c18)
except Exception:
    pass
layout["1I:00AM"] = [523, 893, 764, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/19_text_Earth_Day_Every_Day_Faire_At_Foster.png
try:
    _c19 = get_crop(19, 773, 144)
    canvas.paste(_c19, (144, 1289), _c19)
except Exception:
    pass
layout["Earth_Day_Every_Day_Faire"] = [144, 1289, 917, 1433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/20_text_Foster.png
try:
    _c20 = get_crop(20, 144, 52)
    canvas.paste(_c20, (139, 1566), _c20)
except Exception:
    pass
layout["Foster"] = [139, 1566, 283, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/21_text_Library.png
try:
    _c21 = get_crop(21, 162, 66)
    canvas.paste(_c21, (372, 1561), _c21)
except Exception:
    pass
layout["Library"] = [372, 1561, 534, 1627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/22_text_3_hrs.png
try:
    _c22 = get_crop(22, 112, 50)
    canvas.paste(_c22, (141, 1674), _c22)
except Exception:
    pass
layout["3_hrs"] = [141, 1674, 253, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 63)
    canvas.paste(_c23, (138, 1780), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1517), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 452, 61)
    canvas.paste(_c25, (45, 2080), _c25)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 497, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/26_text_General_Admission.png
try:
    _c26 = get_crop(26, 415, 55)
    canvas.paste(_c26, (116, 2451), _c26)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_10_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-12/27_text_Free.png
try:
    _c27 = get_crop(27, 105, 48)
    canvas.paste(_c27, (116, 2599), _c27)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]
