# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_11
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13.png
# step_index: 11/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile checkout page
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Clear canvas to white (dominant color)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top ~80px) - light gray
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill=(230, 230, 230))

# Header / Toolbar area (just below status bar)
header_top = status_h
header_bottom = 210
draw.rectangle((0, header_top, 1440, header_bottom), fill=(250, 250, 252))

# Header bottom divider (subtle)
draw.line((40, header_bottom, 1400, header_bottom), fill=(227, 226, 232), width=2)

# Subtle shadow under header (thin)
draw.line((0, header_bottom + 2, 1440, header_bottom + 2), fill=(245, 245, 247))

# Main content card behind contact information (rounded rectangle)
card_left = 48
card_top = 280
card_right = 1392
card_bottom = 1520
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=12,
                       fill=(253, 253, 254),
                       outline=(235, 234, 240),
                       width=1)

# Sub-section divider inside the card (above the checkboxes area)
divider_y = 1140
draw.line((card_left + 10, divider_y, card_right - 10, divider_y), fill=(235, 234, 240), width=1)

# Horizontal separator before the register area (light)
sep_y = 1680
draw.line((40, sep_y, 1400, sep_y), fill=(235, 234, 240), width=2)

# Thin divider above "Powered by" area
powered_div_y = 1950
draw.line((40, powered_div_y, 1400, powered_div_y), fill=(235, 234, 240), width=1)

# Light gray footer background strip behind "Powered by" area (subtle)
footer_strip_top = powered_div_y + 12
footer_strip_bottom = powered_div_y + 110
draw.rectangle((40, footer_strip_top, 1400, footer_strip_bottom), fill=(255, 255, 255))

# Large empty content area below powered-by (keeps background consistent)
content_bottom_start = footer_strip_bottom + 20
draw.rectangle((0, content_bottom_start, 1440, 2960), fill=(255, 255, 255))

# Decorative subtle left and right margins (to frame content)
margin_color = (247, 247, 249)
draw.rectangle((0, header_bottom, 40, 2960), fill=margin_color)
draw.rectangle((1400, header_bottom, 1440, 2960), fill=margin_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/00_icon_Last_name.png
try:
    _c0 = get_crop(0, 640, 163)
    canvas.paste(_c0, (736, 540), _c0)
except Exception:
    pass
layout["Last_name*"] = [736, 540, 1376, 703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/01_icon_First_name.png
try:
    _c1 = get_crop(1, 637, 164)
    canvas.paste(_c1, (63, 539), _c1)
except Exception:
    pass
layout["First_name"] = [63, 539, 700, 703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/02_icon_Email_address.png
try:
    _c2 = get_crop(2, 641, 164)
    canvas.paste(_c2, (60, 763), _c2)
except Exception:
    pass
layout["Email_address_*"] = [60, 763, 701, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/03_icon_Confirm_email.png
try:
    _c3 = get_crop(3, 641, 163)
    canvas.paste(_c3, (734, 764), _c3)
except Exception:
    pass
layout["Confirm_email_*"] = [734, 764, 1375, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/04_icon_Register.png
try:
    _c4 = get_crop(4, 1302, 136)
    canvas.paste(_c4, (66, 1758), _c4)
except Exception:
    pass
layout["Register"] = [66, 1758, 1368, 1894]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/05_icon_agree_to_the_Eventbrite_Terms_of_Service.png
try:
    _c5 = get_crop(5, 530, 65)
    canvas.paste(_c5, (727, 1591), _c5)
except Exception:
    pass
layout["agree_to_the_Eventbrite_T"] = [727, 1591, 1257, 1656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/06_icon_4.39.png
try:
    _c6 = get_crop(6, 61, 62)
    canvas.paste(_c6, (179, 2), _c6)
except Exception:
    pass
layout["4.39"] = [179, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 59, 58)
    canvas.paste(_c7, (311, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [311, 4, 370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 56)
    canvas.paste(_c8, (249, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [249, 6, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/09_icon_4.39.png
try:
    _c9 = get_crop(9, 95, 103)
    canvas.paste(_c9, (22, 100), _c9)
except Exception:
    pass
layout["4.39"] = [22, 100, 117, 203]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/10_icon_4.39.png
try:
    _c10 = get_crop(10, 64, 63)
    canvas.paste(_c10, (112, 1), _c10)
except Exception:
    pass
layout["4.39"] = [112, 1, 176, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 48, 58)
    canvas.paste(_c11, (1323, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [1323, 4, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 95, 61)
    canvas.paste(_c12, (1215, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1215, 1, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/13_icon_Powered_by_eventbrite.png
try:
    _c13 = get_crop(13, 421, 66)
    canvas.paste(_c13, (66, 2032), _c13)
except Exception:
    pass
layout["Powered_by_eventbrite"] = [66, 2032, 487, 2098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 79, 73)
    canvas.paste(_c14, (1327, 111), _c14)
except Exception:
    pass
layout["icon_14"] = [1327, 111, 1406, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/15_icon_4.39.png
try:
    _c15 = get_crop(15, 98, 64)
    canvas.paste(_c15, (10, 1), _c15)
except Exception:
    pass
layout["4.39"] = [10, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/16_text_Checkout.png
try:
    _c16 = get_crop(16, 271, 63)
    canvas.paste(_c16, (587, 115), _c16)
except Exception:
    pass
layout["Checkout"] = [587, 115, 858, 178]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/17_text_Time_left_19.55.png
try:
    _c17 = get_crop(17, 292, 50)
    canvas.paste(_c17, (574, 238), _c17)
except Exception:
    pass
layout["Time_left_19.55"] = [574, 238, 866, 288]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/18_text_Contact_information.png
try:
    _c18 = get_crop(18, 673, 79)
    canvas.paste(_c18, (68, 338), _c18)
except Exception:
    pass
layout["Contact_information"] = [68, 338, 741, 417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/19_text_Required.png
try:
    _c19 = get_crop(19, 178, 60)
    canvas.paste(_c19, (93, 457), _c19)
except Exception:
    pass
layout["Required"] = [93, 457, 271, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/20_text_Keep_me_updated_on_more_events_and_news_.png
try:
    _c20 = get_crop(20, 1119, 69)
    canvas.paste(_c20, (178, 1014), _c20)
except Exception:
    pass
layout["Keep_me_updated_on_more_e"] = [178, 1014, 1297, 1083]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/21_text_organizer.png
try:
    _c21 = get_crop(21, 200, 66)
    canvas.paste(_c21, (176, 1084), _c21)
except Exception:
    pass
layout["organizer"] = [176, 1084, 376, 1150]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/22_text_Send_me_emails_about_the_best_events_hap.png
try:
    _c22 = get_crop(22, 1118, 64)
    canvas.paste(_c22, (177, 1201), _c22)
except Exception:
    pass
layout["Send_me_emails_about_the_"] = [177, 1201, 1295, 1265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/23_text_online..png
try:
    _c23 = get_crop(23, 126, 43)
    canvas.paste(_c23, (182, 1273), _c23)
except Exception:
    pass
layout["online."] = [182, 1273, 308, 1316]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/24_text_0_SO.00.png
try:
    _c24 = get_crop(24, 193, 61)
    canvas.paste(_c24, (1169, 1428), _c24)
except Exception:
    pass
layout["0_SO.00"] = [1169, 1428, 1362, 1489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_11_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-13/25_text_By_selecting_Register.png
try:
    _c25 = get_crop(25, 410, 64)
    canvas.paste(_c25, (66, 1593), _c25)
except Exception:
    pass
layout["By_selecting_Register;"] = [66, 1593, 476, 1657]
