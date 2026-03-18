# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_10
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12.png
# step_index: 10/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the Eventbrite-like page
# Assumes 'canvas' (PIL.Image) and 'draw' (PIL.ImageDraw) are provided.

W, H = canvas.size

# 1) Page background (dominant color)
draw.rectangle((0, 0, W, H), fill="#FFFFFF")

# 2) Status bar area at top (~50px)
status_bar_h = 84
draw.rectangle((0, 0, W, status_bar_h), fill="#D0D0D0")

# 3) Header / banner area (dark purple gradient)
banner_top = status_bar_h
banner_bottom = 620
h = banner_bottom - banner_top
c1 = (28, 26, 46)   # top color
c2 = (76, 44, 113)  # bottom color
for i in range(h):
    t = i / max(1, h - 1)
    r = int(c1[0] * (1 - t) + c2[0] * t)
    g = int(c1[1] * (1 - t) + c2[1] * t)
    b = int(c1[2] * (1 - t) + c2[2] * t)
    draw.line((0, banner_top + i, W, banner_top + i), fill=(r, g, b))

# subtle bottom divider under banner
draw.line((48, banner_bottom, W-48, banner_bottom), fill="#0b0b0b", width=1)

# 4) Large content area background (remains white) - add a soft overall subtle tint band below banner
draw.rectangle((0, banner_bottom, W, H), fill="#FFFFFF")

# 5) Organizer / profile card background (rounded rectangle behind organizer row)
org_card_left = 48
org_card_right = W - 48
# Position derived from detected organizer profile at y ~1194
org_card_top = 1182
org_card_bottom = 1332
draw.rounded_rectangle((org_card_left, org_card_top, org_card_right, org_card_bottom),
                       radius=28, fill="#F6F5F9", outline="#E8E6EB", width=1)

# 6) Thin separator under organizer card
sep_y_1 = org_card_bottom + 24
draw.line((org_card_left, sep_y_1, org_card_right, sep_y_1), fill="#EAE8EC", width=2)

# 7) Section divider line (after details/refund policy area)
# This roughly corresponds to the thin line above "About this event"
sep_y_2 = 1728
draw.line((org_card_left, sep_y_2, org_card_right, sep_y_2), fill="#F0EEF2", width=2)

# 8) Another faint divider near the "About this event" heading area
sep_y_3 = 1928
draw.line((org_card_left, sep_y_3, org_card_right, sep_y_3), fill="#F0EEF2", width=1)

# 9) Ticket selection card (outlined rounded rectangle)
# Positioned to contain ticket row and quantity controls (do NOT draw the controls themselves)
ticket_card_top = 2480
ticket_card_bottom = 2624
ticket_card_left = 48
ticket_card_right = W - 48
draw.rounded_rectangle((ticket_card_left, ticket_card_top, ticket_card_right, ticket_card_bottom),
                       radius=24, fill="#FFFFFF", outline="#2F4CFF", width=8)

# subtle inner divider inside ticket card (to separate title/price area)
inner_sep_y = ticket_card_top + 72
draw.line((ticket_card_left + 28, inner_sep_y, ticket_card_right - 28, inner_sep_y), fill="#F0F2FB", width=2)

# 10) Checkout button background (DO NOT draw the text)
checkout_left = 72
checkout_top = 2756
checkout_right = checkout_left + 1296
checkout_bottom = checkout_top + 132
draw.rounded_rectangle((checkout_left, checkout_top, checkout_right, checkout_bottom),
                       radius=12, fill="#C94A20", outline=None)

# 11) Subtle top highlight on checkout button (thin lighter band)
draw.rectangle((checkout_left, checkout_top, checkout_right, checkout_top + 8), fill="#D5602E")

# 12) Subtle shadow under ticket card and checkout button for depth
shadow_color = "#E6E6E6"
draw.rectangle((ticket_card_left, ticket_card_bottom, ticket_card_right, ticket_card_bottom + 6), fill=shadow_color)
draw.rectangle((checkout_left, checkout_bottom, checkout_right, checkout_bottom + 6), fill=shadow_color)

# 13) Final thin page-wide divider near middle (visual structure)
draw.line((36, 1400, W-36, 1400), fill="#F2F1F4", width=1)

# Done. Keep all content elements (icons/text/buttons) to be pasted on top by the pipeline.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1195), _c0)
except Exception:
    pass
layout["Following"] = [946, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/02_icon_Check_out_for_S12.51.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Check_out_for_S12.51"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/04_icon_7.52.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["7.52"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/05_icon_Music.png
try:
    _c5 = get_crop(5, 203, 102)
    canvas.paste(_c5, (41, 2070), _c5)
except Exception:
    pass
layout["Music"] = [41, 2070, 244, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/06_icon_Decrease.png
try:
    _c6 = get_crop(6, 99, 96)
    canvas.paste(_c6, (996, 2444), _c6)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/07_icon_Increase.png
try:
    _c7 = get_crop(7, 96, 96)
    canvas.paste(_c7, (1224, 2444), _c7)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 92, 103)
    canvas.paste(_c8, (1108, 2442), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2442, 1200, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/09_icon_Ticket_sales_end_soon.png
try:
    _c9 = get_crop(9, 548, 85)
    canvas.paste(_c9, (40, 752), _c9)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 752, 588, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 98, 61)
    canvas.paste(_c10, (1216, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 1, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 58, 62)
    canvas.paste(_c11, (1316, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1316, 1, 1374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/12_icon_7.52.png
try:
    _c12 = get_crop(12, 59, 67)
    canvas.paste(_c12, (182, 1), _c12)
except Exception:
    pass
layout["7.52"] = [182, 1, 241, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/13_icon_Jesse_LevIt_QUARTET.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1116, 108), _c13)
except Exception:
    pass
layout["Jesse_LevIt_QUARTET"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/14_icon_7.52.png
try:
    _c14 = get_crop(14, 60, 68)
    canvas.paste(_c14, (115, 0), _c14)
except Exception:
    pass
layout["7.52"] = [115, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 66)
    canvas.paste(_c15, (248, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [248, 2, 301, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 63, 65)
    canvas.paste(_c16, (310, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [310, 2, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/17_icon_S10.00.png
try:
    _c17 = get_crop(17, 75, 72)
    canvas.paste(_c17, (306, 2588), _c17)
except Exception:
    pass
layout["S10.00"] = [306, 2588, 381, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/18_text_7.52.png
try:
    _c18 = get_crop(18, 89, 43)
    canvas.paste(_c18, (22, 17), _c18)
except Exception:
    pass
layout["7.52"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/19_text_Wednesday_April_24.png
try:
    _c19 = get_crop(19, 414, 144)
    canvas.paste(_c19, (288, 1155), _c19)
except Exception:
    pass
layout["Wednesday;_April_24"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/20_text_7_00_PM.png
try:
    _c20 = get_crop(20, 207, 56)
    canvas.paste(_c20, (585, 893), _c20)
except Exception:
    pass
layout["7:00_PM"] = [585, 893, 792, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/21_text_Art_Rangers_Jazz_Night_Auction.png
try:
    _c21 = get_crop(21, 414, 144)
    canvas.paste(_c21, (288, 1155), _c21)
except Exception:
    pass
layout["Art_Rangers_Jazz_Night_&_"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/22_text_Escape_the_Routine.png
try:
    _c22 = get_crop(22, 414, 144)
    canvas.paste(_c22, (288, 1155), _c22)
except Exception:
    pass
layout["Escape_the_Routine"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/23_text_178_Followers.png
try:
    _c23 = get_crop(23, 414, 144)
    canvas.paste(_c23, (288, 1155), _c23)
except Exception:
    pass
layout["178_Followers"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/24_text_The_Faight_Collective.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1422), _c24)
except Exception:
    pass
layout["The_Faight_Collective"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/25_text_3_hrs.png
try:
    _c25 = get_crop(25, 112, 49)
    canvas.paste(_c25, (141, 1580), _c25)
except Exception:
    pass
layout["3_hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/26_text_Refund_policy.png
try:
    _c26 = get_crop(26, 299, 63)
    canvas.paste(_c26, (138, 1685), _c26)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/27_text_The_organizer_will_review_refund_request.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1422), _c27)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/28_text_About_this_event.png
try:
    _c28 = get_crop(28, 455, 65)
    canvas.paste(_c28, (44, 1982), _c28)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 499, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/29_text_Join_us_at_The_Faight_Collective_for_the.png
try:
    _c29 = get_crop(29, 99, 96)
    canvas.paste(_c29, (996, 2444), _c29)
except Exception:
    pass
layout["Join_us_at_The_Faight_Col"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/30_text_General_Admission.png
try:
    _c30 = get_crop(30, 75, 72)
    canvas.paste(_c30, (306, 2588), _c30)
except Exception:
    pass
layout["General_Admission"] = [306, 2588, 381, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/31_text_S10.00.png
try:
    _c31 = get_crop(31, 163, 57)
    canvas.paste(_c31, (113, 2592), _c31)
except Exception:
    pass
layout["S10.00"] = [113, 2592, 276, 2649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_10_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-12/32_clickable_Organizer_profile_picture.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (96, 1194), _c32)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1194, 240, 1338]
