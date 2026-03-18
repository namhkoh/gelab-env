# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_14
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16.png
# step_index: 14/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for mobile UI page (1440x2960)
# Available variables:
# - canvas: PIL.Image (1440x2960 RGB, starts as white)
# - draw: PIL.ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Fill overall background (subtle off-white to match UI)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (approx. 64px high) - dark to contrast status icons
status_h = 64
draw.rectangle((0, 0, 1440, status_h), fill="#263238")

# Header / hero banner (deep blue area for event artwork)
hero_top = status_h
hero_bottom = 420
draw.rectangle((0, hero_top, 1440, hero_bottom), fill="#0d47a1")

# Slight curved-bottom effect for hero: draw a subtle arc shadow to separate from content
# Use a light translucent band - approximate with a thin lighter line
for i, alpha in enumerate([0, 1, 2, 3]):
    y = hero_bottom + i
    draw.line((48, y, 1392, y), fill="#e9eef6" if i >= 2 else "#e6ecf6", width=1)

# Main content area remains white; draw a subtle top margin shadow under the hero
draw.line((48, hero_bottom + 6, 1392, hero_bottom + 6), fill="#eef3f8", width=1)

# Organizer card (rounded rectangle) behind organizer info + follow button
# Keep it lightly tinted and avoid drawing any icons/text or buttons themselves.
org_card_x0 = 48
org_card_x1 = 1392
org_card_y0 = 1000
org_card_y1 = 1168
org_radius = 28
draw.rounded_rectangle((org_card_x0, org_card_y0, org_card_x1, org_card_y1),
                       radius=org_radius, fill="#f6f7fb", outline=None)

# Subtle inner highlight on organizer card (top inner band)
draw.rounded_rectangle((org_card_x0 + 2, org_card_y0 + 2, org_card_x1 - 2, org_card_y0 + 46),
                       radius=org_radius - 2, fill="#fbfbfd", outline=None)

# Soft divider/shadow below the organizer card
draw.line((org_card_x0 + 6, org_card_y1 + 8, org_card_x1 - 6, org_card_y1 + 8), fill="#eceff3", width=2)

# Section separator under event details (thin subtle divider)
sep_y = 1620
draw.line((48, sep_y, 1392, sep_y), fill="#eceff6", width=2)

# Another subtle divider further down (before tickets/CTA area)
sep_y2 = 2040
draw.line((48, sep_y2, 1392, sep_y2), fill="#f0f3f8", width=2)

# "About this event" section background area (just a subtle grouping container)
about_x0 = 48
about_x1 = 1392
about_y0 = 1688
about_y1 = 1968
draw.rectangle((about_x0, about_y0, about_x1, about_y1), fill="#FFFFFF", outline=None)
# light rounded border to suggest card grouping
draw.rounded_rectangle((about_x0, about_y0, about_x1, about_y1),
                       radius=14, outline="#eef2f7", width=2, fill=None)

# Note: Do NOT draw any icons, labels, buttons, or text elements that will be pasted later.
# Avoid drawing inside the bottom Reserve area (y >= 2324) which will be auto-pasted.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/03_icon_Health_Wellness.png
try:
    _c3 = get_crop(3, 234, 119)
    canvas.paste(_c3, (48, 2205), _c3)
except Exception:
    pass
layout["Health_&_Wellness"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1440, 636)
    canvas.paste(_c4, (0, 2324), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/05_icon_9.42.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["9.42"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 112, 106)
    canvas.paste(_c6, (988, 2440), _c6)
except Exception:
    pass
layout["icon_6"] = [988, 2440, 1100, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 108, 105)
    canvas.paste(_c7, (1215, 2441), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 2441, 1323, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 92, 103)
    canvas.paste(_c8, (1108, 2442), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2442, 1200, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 68)
    canvas.paste(_c9, (1155, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1155, 1, 1203, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 59)
    canvas.paste(_c10, (316, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 4, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/11_icon_Caminata_de_Skm.png
try:
    _c11 = get_crop(11, 234, 119)
    canvas.paste(_c11, (48, 2205), _c11)
except Exception:
    pass
layout["Caminata_de_Skm"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 44, 61)
    canvas.paste(_c12, (1327, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1327, 3, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/13_icon_9.42.png
try:
    _c13 = get_crop(13, 52, 61)
    canvas.paste(_c13, (183, 2), _c13)
except Exception:
    pass
layout["9.42"] = [183, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 61)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 62)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 1, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 98, 66)
    canvas.paste(_c16, (1214, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1214, 0, 1312, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/17_icon_9.42.png
try:
    _c17 = get_crop(17, 52, 64)
    canvas.paste(_c17, (117, 1), _c17)
except Exception:
    pass
layout["9.42"] = [117, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/18_icon_Be_Social_Productions.png
try:
    _c18 = get_crop(18, 467, 144)
    canvas.paste(_c18, (288, 1028), _c18)
except Exception:
    pass
layout["Be_Social_Productions"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/19_icon_EVENT.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1116, 108), _c19)
except Exception:
    pass
layout["EVENT"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/20_text_9.42.png
try:
    _c20 = get_crop(20, 91, 41)
    canvas.paste(_c20, (20, 17), _c20)
except Exception:
    pass
layout["9.42"] = [20, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/21_text_HEALTH.png
try:
    _c21 = get_crop(21, 310, 90)
    canvas.paste(_c21, (400, 159), _c21)
except Exception:
    pass
layout["HEALTH"] = [400, 159, 710, 249]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/22_text_Saturday_March_30_._8.00_AM.png
try:
    _c22 = get_crop(22, 467, 144)
    canvas.paste(_c22, (288, 1028), _c22)
except Exception:
    pass
layout["Saturday;_March_30_._8.00"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/23_text_UNIVISION_SK_Walk.png
try:
    _c23 = get_crop(23, 467, 144)
    canvas.paste(_c23, (288, 1028), _c23)
except Exception:
    pass
layout["UNIVISION_SK_Walk"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/24_text_Health_Fair_ELAC.png
try:
    _c24 = get_crop(24, 331, 144)
    canvas.paste(_c24, (1013, 1068), _c24)
except Exception:
    pass
layout["Health_Fair_ELAC"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/25_text_East_Los_Angeles_College.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1295), _c25)
except Exception:
    pass
layout["East_Los_Angeles_College"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/26_text_5_hrs.png
try:
    _c26 = get_crop(26, 112, 50)
    canvas.paste(_c26, (141, 1452), _c26)
except Exception:
    pass
layout["5_hrs"] = [141, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/27_text_Refund_policy.png
try:
    _c27 = get_crop(27, 299, 63)
    canvas.paste(_c27, (138, 1558), _c27)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/28_text_The_organizer_will_review_refund_request.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1295), _c28)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/29_text_General_Admission.png
try:
    _c29 = get_crop(29, 234, 119)
    canvas.paste(_c29, (48, 2205), _c29)
except Exception:
    pass
layout["General_Admission"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/30_text_Free.png
try:
    _c30 = get_crop(30, 105, 48)
    canvas.paste(_c30, (116, 2599), _c30)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_14_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-16/31_clickable_Organizer_profile_picture.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (96, 1067), _c31)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
