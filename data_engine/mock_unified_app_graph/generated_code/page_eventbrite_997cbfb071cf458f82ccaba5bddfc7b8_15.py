# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_15
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17.png
# step_index: 15/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL Image 1440x2960 RGB, draw: ImageDraw)
w, h = canvas.size

# Colors
status_bar_color = "#9ea79f"    # muted green-gray status bar
notif_bg = "#e9f6ef"            # light mint notification banner
notif_border = "#cfe7d8"
page_bg = "#ffffff"
divider_color = "#e9e9eb"
card1_bg = "#fff5f4"            # pale pink
card2_bg = "#f3f6fb"            # pale blue / lavender
card3_bg = "#eef8f3"            # pale mint
card_border = "#ececec"
accent1 = "#f29a86"             # coral accent for first agenda card
accent2 = "#2e3f84"             # deep blue accent for second
accent3 = "#b6d6c1"             # green accent for third
ticket_border = "#355be8"       # bright blue for ticket card border
shadow_color = "#e6e6e9"
thin_gray = "#f2f2f4"
shadow_offset = 8

# Top status bar
status_h = 96
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# Notification banner under the status bar
notif_top = status_h
notif_h = 120
draw.rectangle((0, notif_top, w, notif_top + notif_h), fill=notif_bg)
# subtle bottom border for the banner
draw.line((24, notif_top + notif_h, w - 24, notif_top + notif_h), fill=notif_border, width=1)

# Main page background (ensure whole canvas)
draw.rectangle((0, notif_top + notif_h, w, h), fill=page_bg)

# Thin divider under header area (approx where content separation occurs)
draw.line((24, notif_top + notif_h + 12, w - 24, notif_top + notif_h + 12), fill=divider_color, width=1)

# Agenda heading area separator (a faint horizontal rule)
agenda_sep_y = 560
draw.line((36, agenda_sep_y, w - 36, agenda_sep_y), fill=divider_color, width=1)

# Agenda cards (three rounded rectangles with left accent bars)
x_margin = 48
card_w = w - 2 * x_margin
card_h = 136
radius = 22
spacing = 40

card1_top = agenda_sep_y + 24
card2_top = card1_top + card_h + spacing
card3_top = card2_top + card_h + spacing

# Card 1 - warm/pink
draw.rounded_rectangle((x_margin, card1_top, x_margin + card_w, card1_top + card_h),
                       radius=radius, fill=card1_bg, outline=card_border, width=1)
# left accent bar for card 1
accent_bar_w = 8
draw.rectangle((x_margin + 20, card1_top + 16, x_margin + 20 + accent_bar_w, card1_top + card_h - 16),
               fill=accent1)

# Card 2 - cool/blue
draw.rounded_rectangle((x_margin, card2_top, x_margin + card_w, card2_top + card_h),
                       radius=radius, fill=card2_bg, outline=card_border, width=1)
# left accent bar for card 2
draw.rectangle((x_margin + 20, card2_top + 16, x_margin + 20 + accent_bar_w, card2_top + card_h - 16),
               fill=accent2)

# Card 3 - mint/green
draw.rounded_rectangle((x_margin, card3_top, x_margin + card_w, card3_top + card_h),
                       radius=radius, fill=card3_bg, outline=card_border, width=1)
# left accent bar for card 3
draw.rectangle((x_margin + 20, card3_top + 16, x_margin + 20 + accent_bar_w, card3_top + card_h - 16),
               fill=accent3)

# Subsection divider under agenda cards
divider_y = card3_top + card_h + 56
draw.line((36, divider_y, w - 36, divider_y), fill=thin_gray, width=2)

# FAQs area subtle section top spacing (no text drawn, only separators)
faqs_top = divider_y + 40
draw.line((36, faqs_top + 200, w - 36, faqs_top + 200), fill=divider_color, width=1)

# Big bottom ticket/select box (above reserve button) with drop shadow and border
ticket_top = 2360
ticket_bottom = 2560
ticket_left = 48
ticket_right = w - 48
ticket_radius = 18

# shadow
draw.rounded_rectangle((ticket_left + shadow_offset, ticket_top + shadow_offset,
                        ticket_right + shadow_offset, ticket_bottom + shadow_offset),
                       radius=ticket_radius, fill=shadow_color)

# main ticket card (white with blue border)
draw.rounded_rectangle((ticket_left, ticket_top, ticket_right, ticket_bottom),
                       radius=ticket_radius, fill="#ffffff", outline=ticket_border, width=6)

# subtle inner divider inside ticket card to indicate content grouping (no text)
inner_div_y = ticket_top + 68
draw.line((ticket_left + 28, inner_div_y, ticket_right - 28, inner_div_y), fill=divider_color, width=1)

# Final thin top divider above the reserve button area (so the reserve button will be pasted on top)
reserve_div_y = 2728
draw.line((24, reserve_div_y, w - 24, reserve_div_y), fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/00_icon_Welcome_Real_Estate_Market_Update.png
try:
    _c0 = get_crop(0, 1440, 312)
    canvas.paste(_c0, (0, 0), _c0)
except Exception:
    pass
layout["Welcome_&_Real_Estate_Mar"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/01_icon_Reserve_a_spot.png
try:
    _c1 = get_crop(1, 1296, 132)
    canvas.paste(_c1, (72, 2756), _c1)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/02_icon_The_VA_Home_Loan_Explained_How_to.png
try:
    _c2 = get_crop(2, 1257, 322)
    canvas.paste(_c2, (90, 1017), _c2)
except Exception:
    pass
layout["The_VA_Home_Loan_Explaine"] = [90, 1017, 1347, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2444), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 96, 104)
    canvas.paste(_c5, (1107, 2441), _c5)
except Exception:
    pass
layout["icon_5"] = [1107, 2441, 1203, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/06_icon_9.17.png
try:
    _c6 = get_crop(6, 55, 62)
    canvas.paste(_c6, (114, 3), _c6)
except Exception:
    pass
layout["9.17"] = [114, 3, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/07_icon_Dismiss_notification.png
try:
    _c7 = get_crop(7, 142, 142)
    canvas.paste(_c7, (1251, 97), _c7)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 61)
    canvas.paste(_c8, (314, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [314, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 54)
    canvas.paste(_c9, (1319, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 5, 1371, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/10_icon_9.17.png
try:
    _c10 = get_crop(10, 55, 59)
    canvas.paste(_c10, (181, 3), _c10)
except Exception:
    pass
layout["9.17"] = [181, 3, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 60)
    canvas.paste(_c11, (1210, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1210, 1, 1306, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/12_icon_Free.png
try:
    _c12 = get_crop(12, 147, 112)
    canvas.paste(_c12, (90, 2567), _c12)
except Exception:
    pass
layout["Free"] = [90, 2567, 237, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/13_icon_Free.png
try:
    _c13 = get_crop(13, 75, 72)
    canvas.paste(_c13, (249, 2588), _c13)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 58, 60)
    canvas.paste(_c14, (244, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [244, 3, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/15_icon_Read_less.png
try:
    _c15 = get_crop(15, 238, 65)
    canvas.paste(_c15, (58, 157), _c15)
except Exception:
    pass
layout["Read_less"] = [58, 157, 296, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 42, 53)
    canvas.paste(_c16, (1272, 6), _c16)
except Exception:
    pass
layout["icon_16"] = [1272, 6, 1314, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/17_icon_Where_is_the_event_located.png
try:
    _c17 = get_crop(17, 1248, 204)
    canvas.paste(_c17, (96, 2087), _c17)
except Exception:
    pass
layout["Where_is_the_event_locate"] = [96, 2087, 1344, 2291]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/18_text_9.17.png
try:
    _c18 = get_crop(18, 91, 43)
    canvas.paste(_c18, (20, 17), _c18)
except Exception:
    pass
layout["9.17"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/19_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c19 = get_crop(19, 238, 65)
    canvas.paste(_c19, (58, 157), _c19)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [58, 157, 296, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/20_text_Read_less.png
try:
    _c20 = get_crop(20, 206, 128)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["Read_less"] = [48, 264, 254, 392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/21_text_Agenda.png
try:
    _c21 = get_crop(21, 228, 77)
    canvas.paste(_c21, (44, 509), _c21)
except Exception:
    pass
layout["Agenda"] = [44, 509, 272, 586]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/22_text_7_00_PM.png
try:
    _c22 = get_crop(22, 168, 45)
    canvas.paste(_c22, (216, 1428), _c22)
except Exception:
    pass
layout["7:00_PM"] = [216, 1428, 384, 1473]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/23_text_Q_A.png
try:
    _c23 = get_crop(23, 147, 65)
    canvas.paste(_c23, (217, 1497), _c23)
except Exception:
    pass
layout["Q_&_A"] = [217, 1497, 364, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/24_text_VA.png
try:
    _c24 = get_crop(24, 66, 49)
    canvas.paste(_c24, (118, 2454), _c24)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_15_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-17/25_text_Homebuyer_Webinar.png
try:
    _c25 = get_crop(25, 75, 72)
    canvas.paste(_c25, (249, 2588), _c25)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]
