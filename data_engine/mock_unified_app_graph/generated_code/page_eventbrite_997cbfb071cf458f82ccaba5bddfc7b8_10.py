# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_10
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12.png
# step_index: 10/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw status bar, header, section backgrounds, cards, and separators for the UI mock
# Uses provided: canvas (1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors (approximate from screenshot)
BG = "#FFFFFF"            # main background (white)
STATUS_BAR = "#BFBFBF"    # top status bar gray
HEADER_BG = "#FFFFFF"     # header area (white)
DIVIDER = "#E9E8ED"       # subtle divider lines
CARD1_FILL = "#FFF1F0"    # pale pink card
CARD2_FILL = "#F1F6FB"    # pale blue card
CARD3_FILL = "#F2FBF6"    # pale mint card
CARD_BORDER = "#F0EFF2"   # subtle card border
ACCENT1 = "#F49B88"       # peach accent line (card1)
ACCENT2 = "#2F4AA0"       # navy accent line (card2)
ACCENT3 = "#8DBDA8"       # green accent line (card3)
SUBSECTION_BG = "#FFFFFF" # sections remain white

w, h = canvas.size

# Fill overall background (ensure consistent base)
draw.rectangle([(0,0),(w,h)], fill=BG)

# 1) Status bar area at top (~100px tall)
status_h = 100
draw.rectangle([(0,0),(w,status_h)], fill=STATUS_BAR)

# 2) Header/toolbar area below status bar
header_top = status_h
header_bottom = 240
draw.rectangle([(0,header_top),(w,header_bottom)], fill=HEADER_BG)

# subtle bottom divider under header
draw.line([(48, header_bottom+2), (w-48, header_bottom+2)], fill=DIVIDER, width=2)

# 3) Light informational divider under the short organizer note area
# (this separates header area from main content)
organizer_div_y = header_bottom + 84
draw.line([(48, organizer_div_y), (w-48, organizer_div_y)], fill=DIVIDER, width=1)

# 4) "About this event" section background area (keeps white, but add spacing divider)
about_top = organizer_div_y + 24
about_bottom = about_top + 220
# Add a faint divider below the about summary (to visually separate from Agenda)
draw.line([(48, about_bottom), (w-48, about_bottom)], fill=DIVIDER, width=1)

# 5) Agenda title area spacing (no text)
agenda_y = about_bottom + 30

# 6) Agenda cards: three rounded rectangles with different fills and left accent strokes
card_left = 48
card_right = w - 48
card_width = card_right - card_left
card_height = 140
card_spacing = 30

# Card 1 (pale pink)
c1_top = agenda_y + 60
c1_bottom = c1_top + card_height
draw.rounded_rectangle([(card_left, c1_top), (card_right, c1_bottom)],
                       radius=18, fill=CARD1_FILL, outline=CARD_BORDER, width=1)
# left accent bar for card1
accent_x = card_left + 22
draw.line([(accent_x, c1_top+20), (accent_x, c1_bottom-20)], fill=ACCENT1, width=6)

# Card 2 (pale blue)
c2_top = c1_bottom + card_spacing
c2_bottom = c2_top + card_height
draw.rounded_rectangle([(card_left, c2_top), (card_right, c2_bottom)],
                       radius=18, fill=CARD2_FILL, outline=CARD_BORDER, width=1)
# left accent bar for card2
accent_x2 = card_left + 22
draw.line([(accent_x2, c2_top+20), (accent_x2, c2_bottom-20)], fill=ACCENT2, width=6)

# Card 3 (pale green)
c3_top = c2_bottom + card_spacing
c3_bottom = c3_top + card_height
draw.rounded_rectangle([(card_left, c3_top), (card_right, c3_bottom)],
                       radius=18, fill=CARD3_FILL, outline=CARD_BORDER, width=1)
# left accent bar for card3
accent_x3 = card_left + 22
draw.line([(accent_x3, c3_top+20), (accent_x3, c3_bottom-20)], fill=ACCENT3, width=6)

# 7) Subtle full-width separator above bottom content area (to separate agenda from ticket area)
sep_y = c3_bottom + 60
draw.line([(24, sep_y), (w-24, sep_y)], fill=DIVIDER, width=1)

# 8) Decorative subtle horizontal rule earlier in page (under "Read more" area)
mid_rule_y = about_top + 140
draw.line([(48, mid_rule_y), (w-48, mid_rule_y)], fill=DIVIDER, width=1)

# 9) Small rounded card placeholders to suggest section grouping (no text/icons)
# Top content block (image/media area placeholder) - faint, placed between header and About section
media_top = header_bottom + 18
media_bottom = about_top - 10
if media_bottom - media_top > 30:
    draw.rounded_rectangle([(48, media_top), (w-48, media_bottom)],
                           radius=14, fill="#FFFFFF", outline="#F3F3F5", width=1)

# 10) Subtle left page margin vertical guide (very faint)
draw.line([(48, header_bottom+6), (48, sep_y)], fill="#FCFCFD", width=2)

# Note: do not draw any textual content, icons, or buttons — detected elements will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/01_icon_Welcome_Real_Estate_Market_Update.png
try:
    _c1 = get_crop(1, 234, 144)
    canvas.paste(_c1, (48, 854), _c1)
except Exception:
    pass
layout["Welcome_&_Real_Estate_Mar"] = [48, 854, 282, 998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/03_icon_The_VA_Home_Loan_Explained_How_to.png
try:
    _c3 = get_crop(3, 99, 96)
    canvas.paste(_c3, (996, 2444), _c3)
except Exception:
    pass
layout["The_VA_Home_Loan_Explaine"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1296, 132)
    canvas.paste(_c4, (72, 2756), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/06_icon_Decrease.png
try:
    _c6 = get_crop(6, 99, 96)
    canvas.paste(_c6, (996, 2444), _c6)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/07_icon_9.16.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 108), _c7)
except Exception:
    pass
layout["9.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 94, 102)
    canvas.paste(_c8, (1108, 2442), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2442, 1202, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/09_icon_Free.png
try:
    _c9 = get_crop(9, 75, 72)
    canvas.paste(_c9, (249, 2588), _c9)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 98, 58)
    canvas.paste(_c10, (1216, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 3, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/11_icon_Free.png
try:
    _c11 = get_crop(11, 144, 110)
    canvas.paste(_c11, (93, 2568), _c11)
except Exception:
    pass
layout["Free"] = [93, 2568, 237, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 44, 56)
    canvas.paste(_c12, (1326, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1326, 4, 1370, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/13_icon_9.16.png
try:
    _c13 = get_crop(13, 53, 59)
    canvas.paste(_c13, (183, 2), _c13)
except Exception:
    pass
layout["9.16"] = [183, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/14_icon_Home_Lifestyle.png
try:
    _c14 = get_crop(14, 234, 144)
    canvas.paste(_c14, (48, 854), _c14)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [48, 854, 282, 998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 59)
    canvas.paste(_c15, (315, 4), _c15)
except Exception:
    pass
layout["icon_15"] = [315, 4, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 59)
    canvas.paste(_c16, (247, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [247, 3, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/17_icon_9.16.png
try:
    _c17 = get_crop(17, 52, 61)
    canvas.paste(_c17, (117, 2), _c17)
except Exception:
    pass
layout["9.16"] = [117, 2, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/18_icon_Learn_how_to_use_your_VA_benefits_to_pur.png
try:
    _c18 = get_crop(18, 234, 144)
    canvas.paste(_c18, (48, 854), _c18)
except Exception:
    pass
layout["Learn_how_to_use_your_VA_"] = [48, 854, 282, 998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/19_icon_Active_Military_Ve.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["Active_Military_&_Ve_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/20_text_9.16.png
try:
    _c20 = get_crop(20, 91, 43)
    canvas.paste(_c20, (20, 17), _c20)
except Exception:
    pass
layout["9.16"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/22_text_Agenda.png
try:
    _c22 = get_crop(22, 229, 75)
    canvas.paste(_c22, (42, 1115), _c22)
except Exception:
    pass
layout["Agenda"] = [42, 1115, 271, 1190]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/23_text_7_00_PM.png
try:
    _c23 = get_crop(23, 168, 45)
    canvas.paste(_c23, (216, 2034), _c23)
except Exception:
    pass
layout["7:00_PM"] = [216, 2034, 384, 2079]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/24_text_Q_A.png
try:
    _c24 = get_crop(24, 144, 61)
    canvas.paste(_c24, (219, 2106), _c24)
except Exception:
    pass
layout["Q_&_A"] = [219, 2106, 363, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/25_text_VA.png
try:
    _c25 = get_crop(25, 66, 49)
    canvas.paste(_c25, (118, 2454), _c25)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_10_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-12/26_text_Homebuyer_Webinar.png
try:
    _c26 = get_crop(26, 75, 72)
    canvas.paste(_c26, (249, 2588), _c26)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]
