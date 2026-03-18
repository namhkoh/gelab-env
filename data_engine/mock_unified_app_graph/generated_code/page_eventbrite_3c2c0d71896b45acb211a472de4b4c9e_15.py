# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_15
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17.png
# step_index: 15/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Event page
# Variables provided: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = "#2b2b2b"
banner_color = "#123e8a"       # deep blue banner
banner_shadow = "#0f2e65"
page_bg = "#ffffff"
card_fill = "#f6f7fb"          # very light card background
card_outline = "#ececf6"
divider_color = "#e6e6ea"
ticket_outline = "#2f49d1"     # blue outline for ticket card
ticket_fill = "#ffffff"
subtle_grey = "#f2f3f7"

w, h = canvas.size

# 1) Status bar (top ~60px)
draw.rectangle((0, 0, w, 60), fill=status_bar_color)

# 2) Main banner/header area under status bar
banner_top = 60
banner_bottom = 520
draw.rectangle((0, banner_top, w, banner_bottom), fill=banner_color)

# subtle darker band at the bottom edge of the banner to separate from content
draw.rectangle((0, banner_bottom - 6, w, banner_bottom), fill=banner_shadow)

# 3) Thin divider line just below banner
draw.rectangle((20, banner_bottom, w-20, banner_bottom+2), fill=divider_color)

# 4) Large organizer/info card (rounded) below the banner
org_card_top = banner_bottom + 60
org_card_left = 48
org_card_right = w - 48
org_card_bottom = org_card_top + 160
draw.rounded_rectangle(
    (org_card_left, org_card_top, org_card_right, org_card_bottom),
    radius=22,
    fill=card_fill,
    outline=card_outline,
    width=2
)

# 5) Thin horizontal separator between event details and additional info
sep_y = org_card_bottom + 80
draw.rectangle((40, sep_y, w-40, sep_y+1), fill=divider_color)

# 6) Light section background behind the "About this event" region (subtle)
about_top = sep_y + 36
about_bottom = about_top + 520
draw.rectangle((0, about_top, w, about_bottom), fill=page_bg)  # keep white but ensure separation
# a subtle top divider for the About section
draw.rectangle((40, about_top, w-40, about_top+1), fill=divider_color)

# 7) Another subtle divider after the About area
about_div2 = about_bottom - 20
draw.rectangle((40, about_div2, w-40, about_div2+1), fill=divider_color)

# 8) Ticket selection card (rounded, with blue outline) above the reserve button area
ticket_card_top = about_bottom + 40
ticket_card_bottom = ticket_card_top + 260
draw.rounded_rectangle(
    (48, ticket_card_top, w-48, ticket_card_bottom),
    radius=24,
    fill=ticket_fill,
    outline=ticket_outline,
    width=6
)

# 9) Inner subtle background inside the ticket card for section separation
inner_inset = 18
draw.rounded_rectangle(
    (48 + inner_inset, ticket_card_top + inner_inset, w-48 - inner_inset, ticket_card_top + 120),
    radius=14,
    fill=subtle_grey,
    outline=None
)

# 10) Subtle divider line separating ticket title area from ticket controls area
ctrl_div_y = ticket_card_top + 120 + inner_inset
draw.rectangle((48 + 8, ctrl_div_y, w - 48 - 8, ctrl_div_y + 1), fill=divider_color)

# 11) Large bottom-safe area (leave space for "Reserve a spot" which will be pasted)
# draw a faint backdrop to indicate the footer area but do not draw the Reserve button itself
footer_top = ticket_card_bottom + 40
footer_bottom = h
draw.rectangle((0, footer_top, w, footer_bottom), fill=page_bg)

# 12) Additional separators for visual grouping in the main content column
draw.rectangle((40, org_card_bottom + 40, w-40, org_card_bottom + 41), fill=divider_color)
draw.rectangle((40, about_top + 220, w-40, about_top + 221), fill=divider_color)

# NOTE: No icons, texts, or buttons are drawn here. The detected content (icons, texts, CTA button)
# will be overlaid automatically at their exact positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1068), _c0)
except Exception:
    pass
layout["Following"] = [946, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/03_icon_Health_Wellness.png
try:
    _c3 = get_crop(3, 234, 119)
    canvas.paste(_c3, (48, 2205), _c3)
except Exception:
    pass
layout["Health_&_Wellness"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1296, 132)
    canvas.paste(_c4, (72, 2756), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/05_icon_Decrease.png
try:
    _c5 = get_crop(5, 99, 96)
    canvas.paste(_c5, (996, 2444), _c5)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/06_icon_9.43.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["9.43"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/07_icon_Increase.png
try:
    _c7 = get_crop(7, 96, 96)
    canvas.paste(_c7, (1224, 2444), _c7)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 92, 103)
    canvas.paste(_c8, (1108, 2442), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2442, 1200, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 47, 68)
    canvas.paste(_c9, (1155, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1155, 1, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 60)
    canvas.paste(_c10, (316, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 4, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/11_icon_Join_us_in_the_annual_Health_Fair_5K_Wal.png
try:
    _c11 = get_crop(11, 234, 119)
    canvas.paste(_c11, (48, 2205), _c11)
except Exception:
    pass
layout["Join_us_in_the_annual_Hea"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 44, 62)
    canvas.paste(_c12, (1327, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1327, 3, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/13_icon_9.43.png
try:
    _c13 = get_crop(13, 53, 62)
    canvas.paste(_c13, (183, 1), _c13)
except Exception:
    pass
layout["9.43"] = [183, 1, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 55, 61)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 62)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 1, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 100, 63)
    canvas.paste(_c16, (1214, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [1214, 2, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/17_icon_9.43.png
try:
    _c17 = get_crop(17, 52, 64)
    canvas.paste(_c17, (117, 1), _c17)
except Exception:
    pass
layout["9.43"] = [117, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/18_icon_EVENT.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1116, 108), _c18)
except Exception:
    pass
layout["EVENT"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/19_text_9.43.png
try:
    _c19 = get_crop(19, 91, 41)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.43"] = [20, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/20_text_Saturday_March_30_._8.00_AM.png
try:
    _c20 = get_crop(20, 467, 144)
    canvas.paste(_c20, (288, 1028), _c20)
except Exception:
    pass
layout["Saturday;_March_30_._8.00"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/21_text_UNIVISION_SK_Walk.png
try:
    _c21 = get_crop(21, 467, 144)
    canvas.paste(_c21, (288, 1028), _c21)
except Exception:
    pass
layout["UNIVISION_SK_Walk"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/22_text_Health_Fair_ELAC.png
try:
    _c22 = get_crop(22, 398, 144)
    canvas.paste(_c22, (946, 1068), _c22)
except Exception:
    pass
layout["Health_Fair_ELAC"] = [946, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/23_text_Be_Social_Productions.png
try:
    _c23 = get_crop(23, 467, 144)
    canvas.paste(_c23, (288, 1028), _c23)
except Exception:
    pass
layout["Be_Social_Productions"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/24_text_1.4k_Followers.png
try:
    _c24 = get_crop(24, 467, 144)
    canvas.paste(_c24, (288, 1028), _c24)
except Exception:
    pass
layout["1.4k_Followers"] = [288, 1028, 755, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/25_text_East_Los_Angeles_College.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1295), _c25)
except Exception:
    pass
layout["East_Los_Angeles_College"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/26_text_5_hrs.png
try:
    _c26 = get_crop(26, 112, 50)
    canvas.paste(_c26, (141, 1452), _c26)
except Exception:
    pass
layout["5_hrs"] = [141, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/27_text_Refund_policy.png
try:
    _c27 = get_crop(27, 299, 63)
    canvas.paste(_c27, (138, 1558), _c27)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/28_text_The_organizer_will_review_refund_request.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1295), _c28)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/29_text_General_Admission.png
try:
    _c29 = get_crop(29, 75, 72)
    canvas.paste(_c29, (249, 2588), _c29)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/30_text_Free.png
try:
    _c30 = get_crop(30, 105, 48)
    canvas.paste(_c30, (116, 2599), _c30)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_15_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-17/31_clickable_Organizer_profile_picture.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (96, 1067), _c31)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
