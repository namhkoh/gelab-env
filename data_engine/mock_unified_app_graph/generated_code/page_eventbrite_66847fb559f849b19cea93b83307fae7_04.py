# page_id: page_eventbrite_66847fb559f849b19cea93b83307fae7_04
# screenshot: 2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6.png
# step_index: 4/4
# task: Open Eventbrite. Open favorites and select the second event. Process to checkout and see what payment options it offers.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for checkout page
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw)

# Colors
bg_color = "#FBFBFD"         # page background (very light)
status_color = "#DADADA"     # status bar
header_bg = "#FFFFFF"        # header background
divider_color = "#ECECF0"    # subtle dividers
card_border = "#E6E6EA"      # card border
card_bg = "#FFFFFF"          # card background
shadow_color = "#F6F6F8"     # light shadow / elevated area

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area)
status_h = 92
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Header / toolbar area beneath status bar
header_top = status_h
header_bottom = 192
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)
# subtle bottom divider under header
draw.line([(48, header_bottom), (w-48, header_bottom)], fill=divider_color, width=2)

# Thin hairline shadow under header for depth
draw.line([(0, header_bottom+2), (w, header_bottom+2)], fill=shadow_color, width=1)

# Billing section separation (visual grouping) - light horizontal rule
billing_div_y = 320
draw.line([(48, billing_div_y), (w-48, billing_div_y)], fill=divider_color, width=1)

# "Pay with" section card (rounded rectangle containing payment options)
card_x1, card_x2 = 48, w - 48
card_y1, card_y2 = 1500, 1860
draw.rounded_rectangle(
    [(card_x1, card_y1), (card_x2, card_y2)],
    radius=8,
    fill=card_bg,
    outline=card_border,
    width=2
)
# subtle inner shadow line at top of card
draw.line([(card_x1+2, card_y1+2), (card_x2-2, card_y1+2)], fill=shadow_color, width=1)

# Separator between payment rows inside the card
sep_y = card_y1 + (card_y2 - card_y1) // 2
draw.line([(card_x1+8, sep_y), (card_x2-8, sep_y)], fill=divider_color, width=1)

# Add light divider above the card to separate heading from card
draw.line([(card_x1, card_y1-28), (card_x2, card_y1-28)], fill=divider_color, width=1)

# Price/info area separator lines (above Place Order area)
upper_price_sep = 2330
draw.line([(48, upper_price_sep), (w-48, upper_price_sep)], fill=divider_color, width=1)

# Thin rule above the Place Order button (do not draw the button itself)
place_order_top = 2520
draw.line([(48, place_order_top), (w-48, place_order_top)], fill=divider_color, width=2)

# Bottom area subtle divider near footer ("Powered by" area)
footer_sep_y = h - 140
draw.line([(48, footer_sep_y), (w-48, footer_sep_y)], fill=divider_color, width=1)

# Small decorative horizontal rule near very bottom
draw.line([(48, footer_sep_y+60), (w-48, footer_sep_y+60)], fill=shadow_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/00_icon_Last_name.png
try:
    _c0 = get_crop(0, 640, 164)
    canvas.paste(_c0, (736, 540), _c0)
except Exception:
    pass
layout["Last_name*"] = [736, 540, 1376, 704]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/01_icon_Confirm_email.png
try:
    _c1 = get_crop(1, 642, 163)
    canvas.paste(_c1, (734, 764), _c1)
except Exception:
    pass
layout["Confirm_email_*"] = [734, 764, 1376, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/02_icon_Email_address.png
try:
    _c2 = get_crop(2, 640, 164)
    canvas.paste(_c2, (61, 763), _c2)
except Exception:
    pass
layout["Email_address_*"] = [61, 763, 701, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/03_icon_First_name.png
try:
    _c3 = get_crop(3, 638, 164)
    canvas.paste(_c3, (63, 539), _c3)
except Exception:
    pass
layout["First_name"] = [63, 539, 701, 703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/04_icon_Place_Order.png
try:
    _c4 = get_crop(4, 1301, 139)
    canvas.paste(_c4, (68, 2560), _c4)
except Exception:
    pass
layout["Place_Order"] = [68, 2560, 1369, 2699]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/05_icon_Credit_or_debit_card.png
try:
    _c5 = get_crop(5, 1280, 271)
    canvas.paste(_c5, (78, 1841), _c5)
except Exception:
    pass
layout["Credit_or_debit_card"] = [78, 1841, 1358, 2112]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/06_icon_agree_to_the_Eventbrite_Terms_of_Service.png
try:
    _c6 = get_crop(6, 542, 64)
    canvas.paste(_c6, (790, 2392), _c6)
except Exception:
    pass
layout["agree_to_the_Eventbrite_T"] = [790, 2392, 1332, 2456]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/07_icon_7.38.png
try:
    _c7 = get_crop(7, 62, 62)
    canvas.paste(_c7, (179, 2), _c7)
except Exception:
    pass
layout["7.38"] = [179, 2, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 59, 59)
    canvas.paste(_c8, (311, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [311, 4, 370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 58)
    canvas.paste(_c9, (249, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [249, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/10_icon_Credit_or_debit_card.png
try:
    _c10 = get_crop(10, 1282, 277)
    canvas.paste(_c10, (67, 1568), _c10)
except Exception:
    pass
layout["Credit_or_debit_card"] = [67, 1568, 1349, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/11_icon_7.38.png
try:
    _c11 = get_crop(11, 65, 65)
    canvas.paste(_c11, (111, 1), _c11)
except Exception:
    pass
layout["7.38"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/12_icon_7.38.png
try:
    _c12 = get_crop(12, 100, 123)
    canvas.paste(_c12, (20, 92), _c12)
except Exception:
    pass
layout["7.38"] = [20, 92, 120, 215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 57, 63)
    canvas.paste(_c13, (1316, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1316, 1, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 99, 62)
    canvas.paste(_c14, (1214, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 75, 79)
    canvas.paste(_c15, (114, 1669), _c15)
except Exception:
    pass
layout["icon_15"] = [114, 1669, 189, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 79, 73)
    canvas.paste(_c16, (1327, 111), _c16)
except Exception:
    pass
layout["icon_16"] = [1327, 111, 1406, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/17_icon_PayPal.png
try:
    _c17 = get_crop(17, 73, 79)
    canvas.paste(_c17, (114, 1938), _c17)
except Exception:
    pass
layout["PayPal"] = [114, 1938, 187, 2017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 144, 169)
    canvas.paste(_c18, (1188, 1628), _c18)
except Exception:
    pass
layout["icon_18"] = [1188, 1628, 1332, 1797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/19_icon_S35.00.png
try:
    _c19 = get_crop(19, 121, 87)
    canvas.paste(_c19, (1201, 1928), _c19)
except Exception:
    pass
layout["S35.00"] = [1201, 1928, 1322, 2015]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/20_text_Checkout.png
try:
    _c20 = get_crop(20, 271, 63)
    canvas.paste(_c20, (587, 115), _c20)
except Exception:
    pass
layout["Checkout"] = [587, 115, 858, 178]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/21_text_Time_left_19.54.png
try:
    _c21 = get_crop(21, 292, 50)
    canvas.paste(_c21, (574, 238), _c21)
except Exception:
    pass
layout["Time_left_19.54"] = [574, 238, 866, 288]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/22_text_Billing_information.png
try:
    _c22 = get_crop(22, 631, 97)
    canvas.paste(_c22, (63, 336), _c22)
except Exception:
    pass
layout["Billing_information"] = [63, 336, 694, 433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/23_text_Required.png
try:
    _c23 = get_crop(23, 178, 60)
    canvas.paste(_c23, (93, 457), _c23)
except Exception:
    pass
layout["Required"] = [93, 457, 271, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/24_text_Keep_me_updated_on_more_events_and_news_.png
try:
    _c24 = get_crop(24, 1119, 69)
    canvas.paste(_c24, (178, 1014), _c24)
except Exception:
    pass
layout["Keep_me_updated_on_more_e"] = [178, 1014, 1297, 1083]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/25_text_organizer.png
try:
    _c25 = get_crop(25, 200, 66)
    canvas.paste(_c25, (176, 1084), _c25)
except Exception:
    pass
layout["organizer"] = [176, 1084, 376, 1150]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/26_text_Send_me_emails_about_the_best_events_hap.png
try:
    _c26 = get_crop(26, 1118, 64)
    canvas.paste(_c26, (177, 1201), _c26)
except Exception:
    pass
layout["Send_me_emails_about_the_"] = [177, 1201, 1295, 1265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/27_text_online..png
try:
    _c27 = get_crop(27, 124, 43)
    canvas.paste(_c27, (184, 1273), _c27)
except Exception:
    pass
layout["online."] = [184, 1273, 308, 1316]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/28_text_Pay_with.png
try:
    _c28 = get_crop(28, 302, 101)
    canvas.paste(_c28, (60, 1429), _c28)
except Exception:
    pass
layout["Pay_with"] = [60, 1429, 362, 1530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/29_text_S35.00.png
try:
    _c29 = get_crop(29, 165, 61)
    canvas.paste(_c29, (1197, 2228), _c29)
except Exception:
    pass
layout["S35.00"] = [1197, 2228, 1362, 2289]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/30_text_By_selecting_Place_Order.png
try:
    _c30 = get_crop(30, 473, 60)
    canvas.paste(_c30, (66, 2396), _c30)
except Exception:
    pass
layout["By_selecting_Place_Order;"] = [66, 2396, 539, 2456]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_04_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-6/31_text_Powered_by_eventbrite.png
try:
    _c31 = get_crop(31, 421, 54)
    canvas.paste(_c31, (70, 2842), _c31)
except Exception:
    pass
layout["Powered_by_eventbrite"] = [70, 2842, 491, 2896]
