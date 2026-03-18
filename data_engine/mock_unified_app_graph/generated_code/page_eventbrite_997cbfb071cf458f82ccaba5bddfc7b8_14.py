# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_14
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16.png
# step_index: 14/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_height = 60
draw.rectangle([(0, 0), (1440, status_height)], fill="#d1d1d1")

# Header area (toolbar) with subtle bottom divider
header_top = status_height
header_height = 96
draw.rectangle([(0, header_top), (1440, header_top + header_height)], fill="#ffffff")
draw.line([(24, header_top + header_height - 1), (1416, header_top + header_height - 1)], fill="#efeef2", width=2)

# Page background (slightly warm white)
draw.rectangle([(0, header_top + header_height), (1440, 2960)], fill="#ffffff")

# "Agenda" section top spacing area (leave text/icons out)
agenda_start_y = 500

# Draw three agenda item cards (rounded rectangles with left accent bars)
card_x1 = 48
card_x2 = 1440 - 48
card_width = card_x2 - card_x1
card_height = 150
card_radius = 18
card_spacing = 30

# Card 1 - warm/pink background with coral accent
c1_y1 = agenda_start_y + 80
c1_y2 = c1_y1 + card_height
draw.rounded_rectangle([(card_x1, c1_y1), (card_x2, c1_y2)], radius=card_radius, fill="#fff5f4")
# left accent bar
accent_w = 8
accent_pad = 36
draw.rounded_rectangle([(card_x1 + accent_pad, c1_y1 + 18), (card_x1 + accent_pad + accent_w, c1_y2 - 18)], radius=4, fill="#ffb4a3")

# Card 2 - cool/blue background with navy accent
c2_y1 = c1_y2 + card_spacing
c2_y2 = c2_y1 + card_height
draw.rounded_rectangle([(card_x1, c2_y1), (card_x2, c2_y2)], radius=card_radius, fill="#f3f6fb")
draw.rounded_rectangle([(card_x1 + accent_pad, c2_y1 + 18), (card_x1 + accent_pad + accent_w, c2_y2 - 18)], radius=4, fill="#314a78")

# Card 3 - mint/green background with sage accent
c3_y1 = c2_y2 + card_spacing
c3_y2 = c3_y1 + card_height
draw.rounded_rectangle([(card_x1, c3_y1), (card_x2, c3_y2)], radius=card_radius, fill="#f4fbf7")
draw.rounded_rectangle([(card_x1 + accent_pad, c3_y1 + 18), (card_x1 + accent_pad + accent_w, c3_y2 - 18)], radius=4, fill="#9fcfb8")

# Separator line below agenda/cards
sep_y = c3_y2 + 56
draw.line([(48, sep_y), (1392, sep_y)], fill="#efeef2", width=2)

# FAQs header area (leave text out) with bottom divider
faqs_y = sep_y + 40
draw.rectangle([(0, faqs_y), (1440, faqs_y + 120)], fill="#ffffff")
draw.line([(48, faqs_y + 118), (1392, faqs_y + 118)], fill="#f0eef3", width=1)

# Light horizontal rule under FAQ question area
faq_rule_y = faqs_y + 140
draw.line([(48, faq_rule_y), (1392, faq_rule_y)], fill="#efeef2", width=2)

# Ticket selection container (rounded bordered box)
ticket_x1 = 48
ticket_x2 = 1392
ticket_y1 = 2360
ticket_y2 = 2520
ticket_radius = 16
border_color = "#334dff"
# outer border
draw.rounded_rectangle([(ticket_x1, ticket_y1), (ticket_x2, ticket_y2)], radius=ticket_radius, outline=border_color, width=6, fill="#ffffff")
# slight inner divider
draw.line([(ticket_x1 + 24, ticket_y1 + 86), (ticket_x2 - 24, ticket_y1 + 86)], fill="#f0eef3", width=1)

# Small subtle drop shadow under the ticket box
shadow_y1 = ticket_y2
shadow_y2 = ticket_y2 + 12
for i in range(6):
    alpha = int(12 - i*2)
    if alpha <= 0:
        continue
    # simulate shadow by drawing translucent gray lines (PIL ImageDraw in RGB - approximate with light gray)
    draw.line([(ticket_x1 + 4 + i, shadow_y1 + i), (ticket_x2 - 4 - i, shadow_y1 + i)], fill="#e9e9ef", width=1)

# Bottom safe area/background behind the primary CTA (do not draw CTA button itself)
bottom_safe_y = 2700
draw.rectangle([(0, bottom_safe_y), (1440, 2960)], fill="#ffffff")
# subtle top border above CTA area
draw.line([(48, bottom_safe_y), (1392, bottom_safe_y)], fill="#efeef2", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/00_icon_Welcome_Real_Estate_Market_Update.png
try:
    _c0 = get_crop(0, 206, 128)
    canvas.paste(_c0, (48, 264), _c0)
except Exception:
    pass
layout["Welcome_&_Real_Estate_Mar"] = [48, 264, 254, 392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/02_icon_The_VA_Home_Loan_Explained_How_to.png
try:
    _c2 = get_crop(2, 1257, 322)
    canvas.paste(_c2, (90, 1017), _c2)
except Exception:
    pass
layout["The_VA_Home_Loan_Explaine"] = [90, 1017, 1347, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/04_icon_9.17.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["9.17"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/05_icon_Reserve_a_spot.png
try:
    _c5 = get_crop(5, 1296, 132)
    canvas.paste(_c5, (72, 2756), _c5)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/06_icon_Increase.png
try:
    _c6 = get_crop(6, 96, 96)
    canvas.paste(_c6, (1224, 2444), _c6)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/07_icon_Decrease.png
try:
    _c7 = get_crop(7, 99, 96)
    canvas.paste(_c7, (996, 2444), _c7)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 94, 103)
    canvas.paste(_c8, (1108, 2441), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2441, 1202, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 99, 59)
    canvas.paste(_c9, (1214, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1214, 1, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 58)
    canvas.paste(_c10, (1318, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [1318, 3, 1372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/11_icon_9.17.png
try:
    _c11 = get_crop(11, 53, 61)
    canvas.paste(_c11, (183, 1), _c11)
except Exception:
    pass
layout["9.17"] = [183, 1, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 59)
    canvas.paste(_c12, (315, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [315, 4, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/13_icon_9.17.png
try:
    _c13 = get_crop(13, 52, 62)
    canvas.paste(_c13, (116, 1), _c13)
except Exception:
    pass
layout["9.17"] = [116, 1, 168, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/14_icon_Free.png
try:
    _c14 = get_crop(14, 75, 72)
    canvas.paste(_c14, (249, 2588), _c14)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/15_icon_Free.png
try:
    _c15 = get_crop(15, 146, 114)
    canvas.paste(_c15, (91, 2566), _c15)
except Exception:
    pass
layout["Free"] = [91, 2566, 237, 2680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/16_icon_Where_is_the_event_located.png
try:
    _c16 = get_crop(16, 1248, 204)
    canvas.paste(_c16, (96, 2087), _c16)
except Exception:
    pass
layout["Where_is_the_event_locate"] = [96, 2087, 1344, 2291]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/17_icon_The_VA_Home_Loan_Explained_How_to.png
try:
    _c17 = get_crop(17, 1248, 204)
    canvas.paste(_c17, (96, 2087), _c17)
except Exception:
    pass
layout["The_VA_Home_Loan_Explaine"] = [96, 2087, 1344, 2291]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/18_icon_Active_Military_Ve_-.png
try:
    _c18 = get_crop(18, 206, 128)
    canvas.paste(_c18, (48, 264), _c18)
except Exception:
    pass
layout["Active_Military_&_Ve_-"] = [48, 264, 254, 392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 55, 59)
    canvas.paste(_c19, (246, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [246, 3, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/20_icon_Read_less.png
try:
    _c20 = get_crop(20, 206, 128)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["Read_less"] = [48, 264, 254, 392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/21_text_9.17.png
try:
    _c21 = get_crop(21, 89, 43)
    canvas.paste(_c21, (20, 17), _c21)
except Exception:
    pass
layout["9.17"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/22_text_Agenda.png
try:
    _c22 = get_crop(22, 228, 77)
    canvas.paste(_c22, (44, 509), _c22)
except Exception:
    pass
layout["Agenda"] = [44, 509, 272, 586]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/23_text_VA.png
try:
    _c23 = get_crop(23, 66, 49)
    canvas.paste(_c23, (118, 2454), _c23)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_14_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-16/24_text_Homebuyer_Webinar.png
try:
    _c24 = get_crop(24, 75, 72)
    canvas.paste(_c24, (249, 2588), _c24)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]
