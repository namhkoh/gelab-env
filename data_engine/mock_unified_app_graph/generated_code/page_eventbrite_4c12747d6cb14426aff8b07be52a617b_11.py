# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_11
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13.png
# step_index: 11/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for Event page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

W, H = canvas.size

# Base background (dominant white)
draw.rectangle([0, 0, W, H], fill="#FFFFFF")

# Status bar area (top ~96px) - solid muted green/grey like Android status bar
status_h = 96
draw.rectangle([0, 0, W, status_h], fill="#9EA69F")

# Notification banner (mint/green success banner below status bar)
notif_h = 120
notif_y0 = status_h + 8
notif_y1 = notif_y0 + notif_h
# light rounded banner across the width with small margins
notif_margin = 20
draw.rounded_rectangle(
    [notif_margin, notif_y0, W - notif_margin, notif_y1],
    radius=12,
    fill="#E9F7EE",
    outline="#D6EFD9",
    width=1
)
# thin divider under notification
draw.line([notif_margin, notif_y1 + 6, W - notif_margin, notif_y1 + 6], fill="#E6E9E7", width=1)

# Large event banner area (dark image strip under notification)
banner_y0 = notif_y1 + 24
banner_h = 320
banner_y1 = banner_y0 + banner_h
# subtle vertical gradient from deep navy to very dark
top_color = (56, 76, 106)
bottom_color = (16, 18, 20)
for i in range(banner_h):
    r = int(top_color[0] + (bottom_color[0] - top_color[0]) * (i / max(1, banner_h - 1)))
    g = int(top_color[1] + (bottom_color[1] - top_color[1]) * (i / max(1, banner_h - 1)))
    b = int(top_color[2] + (bottom_color[2] - top_color[2]) * (i / max(1, banner_h - 1)))
    draw.line([0, banner_y0 + i, W, banner_y0 + i], fill=(r, g, b))

# A subtle dark overlay band at very top of the banner (to mimic crop vignette)
draw.rectangle([0, banner_y0, W, banner_y0 + 28], fill=(20, 24, 30))

# Main content background remains white; add a faint full-width divider under the banner
divider_y = banner_y1 + 28
draw.line([48, divider_y, W - 48, divider_y], fill="#EFEFF1", width=1)

# Organizer card (rounded rectangle container behind avatar + follow button)
card_left = 48
card_right = W - 48
card_top = 1100
card_bottom = 1288
card_radius = 28
# shadow (subtle, drawn as a slightly offset faint rectangle)
draw.rounded_rectangle(
    [card_left + 6, card_top + 6, card_right + 6, card_bottom + 6],
    radius=card_radius,
    fill="#F2F3F5"
)
# card background
draw.rounded_rectangle(
    [card_left, card_top, card_right, card_bottom],
    radius=card_radius,
    fill="#FBFBFD",
    outline="#E8EAF0",
    width=2
)

# Small horizontal separator lines between information sections
sep1_y = 1600
sep2_y = 1960
draw.line([48, sep1_y, W - 48, sep1_y], fill="#F1F2F4", width=1)
draw.line([48, sep2_y, W - 48, sep2_y], fill="#F1F2F4", width=1)

# "About this event" content block background - keep white but add top padding divider
about_top = 1880
about_bottom = 2170
# soft background band to separate the 'About' area from white page (very subtle)
draw.rectangle([48, about_top, W - 48, about_top + 4], fill="#F7F8FA")

# Ticket selection card (rounded container with prominent blue border)
ticket_left = 48
ticket_right = W - 48
ticket_top = 2250
ticket_bottom = 2460
ticket_radius = 20
# outer border shadow (subtle)
draw.rounded_rectangle([ticket_left + 4, ticket_top + 6, ticket_right + 4, ticket_bottom + 6],
                       radius=ticket_radius, fill="#F6F7F9")
# white card
draw.rounded_rectangle([ticket_left, ticket_top, ticket_right, ticket_bottom],
                       radius=ticket_radius, fill="#FFFFFF",
                       outline="#3757F0", width=6)

# inner subtle divider within ticket card (to separate title and price/controls)
inner_div_y = ticket_top + 120
draw.line([ticket_left + 28, inner_div_y, ticket_right - 28, inner_div_y], fill="#F0F1F5", width=1)

# A faint bounding area above the checkout button to visually separate it (do NOT draw the button itself)
checkout_area_top = 2690
draw.rectangle([48, checkout_area_top - 6, W - 48, checkout_area_top + 6], fill="#FFFFFF")

# Bottom safe area - keep white and add a subtle top hairline
draw.line([0, H - 140, W, H - 140], fill="#F2F3F5", width=1)

# Final subtle vertical rhythm lines (margins) to mimic page gutters
gutter_x = 48
draw.line([gutter_x, banner_y1 + 12, gutter_x, H - 180], fill="#FFFFFF", width=1)
draw.line([W - gutter_x, banner_y1 + 12, W - gutter_x, H - 180], fill="#FFFFFF", width=1)

# Note: All textual and icon content will be pasted on top at detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1195), _c0)
except Exception:
    pass
layout["Following"] = [946, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/01_icon_Check_out_for_S12.51.png
try:
    _c1 = get_crop(1, 1296, 132)
    canvas.paste(_c1, (72, 2756), _c1)
except Exception:
    pass
layout["Check_out_for_S12.51"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/02_icon_Decrease.png
try:
    _c2 = get_crop(2, 99, 96)
    canvas.paste(_c2, (996, 2444), _c2)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/04_icon_Music.png
try:
    _c4 = get_crop(4, 205, 103)
    canvas.paste(_c4, (40, 2070), _c4)
except Exception:
    pass
layout["Music"] = [40, 2070, 245, 2173]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 92, 103)
    canvas.paste(_c5, (1108, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2442, 1200, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 58, 57)
    canvas.paste(_c6, (311, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [311, 5, 369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/07_icon_Dismiss_notification.png
try:
    _c7 = get_crop(7, 142, 142)
    canvas.paste(_c7, (1251, 97), _c7)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 52)
    canvas.paste(_c8, (252, 8), _c8)
except Exception:
    pass
layout["icon_8"] = [252, 8, 298, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/09_icon_7.53.png
try:
    _c9 = get_crop(9, 61, 60)
    canvas.paste(_c9, (180, 2), _c9)
except Exception:
    pass
layout["7.53"] = [180, 2, 241, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/10_icon_Ticket_sales_end_soon.png
try:
    _c10 = get_crop(10, 548, 84)
    canvas.paste(_c10, (40, 753), _c10)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 588, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/11_icon_7.53.png
try:
    _c11 = get_crop(11, 63, 63)
    canvas.paste(_c11, (112, 1), _c11)
except Exception:
    pass
layout["7.53"] = [112, 1, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 60)
    canvas.paste(_c12, (1325, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1325, 4, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 62, 62)
    canvas.paste(_c13, (1212, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 2, 1274, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 46, 63)
    canvas.paste(_c14, (1267, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1267, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 60)
    canvas.paste(_c15, (382, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 3, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/16_icon_JNdset.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (36, 108), _c16)
except Exception:
    pass
layout["JNdset"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/17_icon_Check_out_for_S12.51.png
try:
    _c17 = get_crop(17, 99, 96)
    canvas.paste(_c17, (996, 2444), _c17)
except Exception:
    pass
layout["Check_out_for_S12.51"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/18_text_7.53.png
try:
    _c18 = get_crop(18, 91, 45)
    canvas.paste(_c18, (20, 15), _c18)
except Exception:
    pass
layout["7.53"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/19_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/20_text_Wednesday_April_24.png
try:
    _c20 = get_crop(20, 414, 144)
    canvas.paste(_c20, (288, 1155), _c20)
except Exception:
    pass
layout["Wednesday;_April_24"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/21_text_7_00_PM.png
try:
    _c21 = get_crop(21, 207, 56)
    canvas.paste(_c21, (585, 893), _c21)
except Exception:
    pass
layout["7:00_PM"] = [585, 893, 792, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/22_text_Art_Rangers_Jazz_Night_Auction.png
try:
    _c22 = get_crop(22, 414, 144)
    canvas.paste(_c22, (288, 1155), _c22)
except Exception:
    pass
layout["Art_Rangers_Jazz_Night_&_"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/23_text_Escape_the_Routine.png
try:
    _c23 = get_crop(23, 414, 144)
    canvas.paste(_c23, (288, 1155), _c23)
except Exception:
    pass
layout["Escape_the_Routine"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/24_text_178_Followers.png
try:
    _c24 = get_crop(24, 414, 144)
    canvas.paste(_c24, (288, 1155), _c24)
except Exception:
    pass
layout["178_Followers"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/25_text_The_Faight_Collective.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1422), _c25)
except Exception:
    pass
layout["The_Faight_Collective"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/26_text_3_hrs.png
try:
    _c26 = get_crop(26, 112, 49)
    canvas.paste(_c26, (141, 1580), _c26)
except Exception:
    pass
layout["3_hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/27_text_Refund_policy.png
try:
    _c27 = get_crop(27, 299, 63)
    canvas.paste(_c27, (138, 1685), _c27)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/28_text_The_organizer_will_review_refund_request.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1422), _c28)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/29_text_About_this_event.png
try:
    _c29 = get_crop(29, 455, 65)
    canvas.paste(_c29, (44, 1982), _c29)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 499, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/30_text_Join_us_at_The_Faight_Collective_for_the.png
try:
    _c30 = get_crop(30, 99, 96)
    canvas.paste(_c30, (996, 2444), _c30)
except Exception:
    pass
layout["Join_us_at_The_Faight_Col"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/31_clickable_Organizer_profile_picture.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (96, 1194), _c31)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1194, 240, 1338]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_11_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-13/32_clickable_Show_more_information.png
try:
    _c32 = get_crop(32, 75, 72)
    canvas.paste(_c32, (306, 2588), _c32)
except Exception:
    pass
layout["Show_more_information"] = [306, 2588, 381, 2660]
