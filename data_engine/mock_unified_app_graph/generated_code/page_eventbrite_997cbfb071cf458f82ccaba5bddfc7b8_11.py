# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_11
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13.png
# step_index: 11/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite page mockup.
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw) objects.

# Overall page background (slightly off-white to match app background)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFCFF")

# Status bar area (top ~96px) - subtle gray
STATUS_H = 96
draw.rectangle([(0, 0), (1440, STATUS_H)], fill="#CFCFCF")

# Subtle status bar bottom separator shadow
draw.line([(0, STATUS_H), (1440, STATUS_H)], fill="#BEBEBE", width=1)

# Header / toolbar area (below status bar)
HEADER_H = 240
draw.rectangle([(0, STATUS_H), (1440, HEADER_H)], fill="#FFFFFF")

# Header bottom divider (thin)
draw.line([(48, HEADER_H), (1392, HEADER_H)], fill="#ECECEC", width=2)

# Light horizontal rule under the organizer-refund note area
RULE1_Y = 320
draw.line([(48, RULE1_Y), (1392, RULE1_Y)], fill="#F0F0F0", width=1)

# Secondary subtle divider further down (between content groups)
RULE2_Y = 2320
draw.line([(24, RULE2_Y), (1416, RULE2_Y)], fill="#F2F2F2", width=1)

# Content area background region (main white body) - keep it white but draw a faint card band behind long text area
BODY_LEFT = 48
BODY_RIGHT = 1392
BODY_TOP = HEADER_H + 24
BODY_BOTTOM = RULE2_Y - 24
draw.rectangle([(BODY_LEFT, BODY_TOP), (BODY_RIGHT, BODY_BOTTOM)], fill="#FFFFFF")

# Slight vignette shadow under header to separate visually
shadow_y0 = HEADER_H - 6
shadow_y1 = HEADER_H + 6
for i in range(6):
    alpha = int(12 - i*2)
    if alpha <= 0:
        break
    # draw expanding faint lines
    draw.line([(48, shadow_y0 + i), (1392, shadow_y0 + i)], fill=(230,230,230), width=1)

# Ticket/card area (rounded rectangle with colored outline)
card_x0, card_y0 = 48, 2440 - 80   # top of the ticket card region
card_x1, card_y1 = 1392, 2600      # bottom of the ticket card region
card_radius = 20
outline_color = "#3B4BF0"  # bluish outline similar to Eventbrite accent
# Outer rounded border
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=card_radius, fill="#FFFFFF", outline=outline_color, width=6)
# Inner subtle highlight (to simulate inner white space)
inner_pad = 8
draw.rounded_rectangle([(card_x0+inner_pad, card_y0+inner_pad), (card_x1-inner_pad, card_y1-inner_pad)], radius=card_radius-6, fill="#FFFFFF", outline=None, width=0)

# Small subtle shadow below card
shadow_y = card_y1 + 6
for i in range(4):
    shade = 240 - i*8
    draw.line([(card_x0+6, shadow_y + i), (card_x1-6, shadow_y + i)], fill=(shade,shade,shade), width=1)

# Reserve button (large orange CTA) - positioned near bottom, rounded
cta_x0, cta_x1 = 48, 1392
cta_y0, cta_y1 = 2680, 2830
cta_radius = 12
draw.rounded_rectangle([(cta_x0, cta_y0), (cta_x1, cta_y1)], radius=cta_radius, fill="#C94A24", outline=None)
# Subtle top highlight on CTA
draw.rounded_rectangle([(cta_x0+2, cta_y0+2), (cta_x1-2, cta_y0+18)], radius=cta_radius-6, fill=(220,95,60,40))

# Large content separators (light rules) above and below sections to create grouping
section_lines = [520, 920, 1540, 2020]
for y in section_lines:
    draw.line([(48, y), (1392, y)], fill="#F5F5F6", width=1)

# Decorative faint left and right gutters to match mobile padding
gutter_color = "#FAFAFB"
draw.rectangle([(0, 0), (24, 2960)], fill=gutter_color)
draw.rectangle([(1416, 0), (1440, 2960)], fill=gutter_color)

# Subtle page bottom safe-area fill (slightly darker than white)
draw.rectangle([(0, 2888), (1440, 2960)], fill="#FCF8F6")

# Note: all UI elements like icons and text will be pasted on top of these structural shapes;
# this code purposefully draws only backgrounds, cards, dividers, and CTA background.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/02_icon_9.16.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["9.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2444), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/05_icon_Reserve_a_spot.png
try:
    _c5 = get_crop(5, 1296, 132)
    canvas.paste(_c5, (72, 2756), _c5)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 101)
    canvas.paste(_c6, (1108, 2442), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2442, 1201, 2543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/07_icon_Home_Lifestyle.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 108), _c7)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/08_icon_Free.png
try:
    _c8 = get_crop(8, 139, 104)
    canvas.paste(_c8, (96, 2573), _c8)
except Exception:
    pass
layout["Free"] = [96, 2573, 235, 2677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 58)
    canvas.paste(_c9, (248, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [248, 4, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 57)
    canvas.paste(_c10, (316, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 5, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 68, 65)
    canvas.paste(_c11, (1176, 1117), _c11)
except Exception:
    pass
layout["icon_11"] = [1176, 1117, 1244, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/12_icon_9.16.png
try:
    _c12 = get_crop(12, 51, 58)
    canvas.paste(_c12, (184, 3), _c12)
except Exception:
    pass
layout["9.16"] = [184, 3, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 44, 55)
    canvas.paste(_c13, (1326, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1326, 5, 1370, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/14_icon_Free.png
try:
    _c14 = get_crop(14, 75, 72)
    canvas.paste(_c14, (249, 2588), _c14)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 95, 57)
    canvas.paste(_c15, (1218, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [1218, 3, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/16_icon_9.16.png
try:
    _c16 = get_crop(16, 53, 60)
    canvas.paste(_c16, (117, 3), _c16)
except Exception:
    pass
layout["9.16"] = [117, 3, 170, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/17_icon_Active_Military_Ve.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (36, 108), _c17)
except Exception:
    pass
layout["Active_Military_&_Ve_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 47, 60)
    canvas.paste(_c18, (384, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [384, 3, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/19_text_9.16.png
try:
    _c19 = get_crop(19, 91, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.16"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/20_text_The_organizer_will_review_refund_request.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1116, 108), _c20)
except Exception:
    pass
layout["The_organizer_will_review"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/21_text_About_this_event.png
try:
    _c21 = get_crop(21, 453, 65)
    canvas.paste(_c21, (44, 504), _c21)
except Exception:
    pass
layout["About_this_event"] = [44, 504, 497, 569]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/22_text_Learn_how_to_use_your_VA_benefits_to_pur.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1116, 108), _c22)
except Exception:
    pass
layout["Learn_how_to_use_your_VA_"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/23_text_down_and_qualify_using_your_BAH.png
try:
    _c23 = get_crop(23, 708, 63)
    canvas.paste(_c23, (41, 806), _c23)
except Exception:
    pass
layout["down_and_qualify_using_yo"] = [41, 806, 749, 869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/24_text_Calling_all_Active_Military_and_Veterans.png
try:
    _c24 = get_crop(24, 1090, 73)
    canvas.paste(_c24, (108, 926), _c24)
except Exception:
    pass
layout["Calling_all_Active_Milita"] = [108, 926, 1198, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/25_text_Ready_to_unlock_the.png
try:
    _c25 = get_crop(25, 425, 63)
    canvas.paste(_c25, (42, 1057), _c25)
except Exception:
    pass
layout["Ready_to_unlock_the"] = [42, 1057, 467, 1120]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/26_text_Discover_the_unbeatable_benefits_of_the_.png
try:
    _c26 = get_crop(26, 1349, 66)
    canvas.paste(_c26, (43, 1245), _c26)
except Exception:
    pass
layout["Discover_the_unbeatable_b"] = [43, 1245, 1392, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/27_text_how_you_can_take_advantage_of_low-intere.png
try:
    _c27 = get_crop(27, 1236, 64)
    canvas.paste(_c27, (43, 1373), _c27)
except Exception:
    pass
layout["how_you_can_take_advantag"] = [43, 1373, 1279, 1437]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/28_text_your_dream_of_owning_a_home_into_a_reali.png
try:
    _c28 = get_crop(28, 900, 62)
    canvas.paste(_c28, (43, 1499), _c28)
except Exception:
    pass
layout["your_dream_of_owning_a_ho"] = [43, 1499, 943, 1561]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/29_text_Our_webinar_will_dive_into_local_insight.png
try:
    _c29 = get_crop(29, 99, 96)
    canvas.paste(_c29, (996, 2444), _c29)
except Exception:
    pass
layout["Our_webinar_will_dive_int"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/30_text_How_to_purchase_a_home_with_Zero_down_VA.png
try:
    _c30 = get_crop(30, 1034, 65)
    canvas.paste(_c30, (69, 1811), _c30)
except Exception:
    pass
layout["How_to_purchase_a_home_wi"] = [69, 1811, 1103, 1876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/31_text_How_to_get_Seller_Paid_Closing_Costs.png
try:
    _c31 = get_crop(31, 768, 68)
    canvas.paste(_c31, (69, 1938), _c31)
except Exception:
    pass
layout["How_to_get_Seller_Paid_Cl"] = [69, 1938, 837, 2006]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/32_text_How_to_use_your_BAH_to_qualify.png
try:
    _c32 = get_crop(32, 653, 64)
    canvas.paste(_c32, (66, 2067), _c32)
except Exception:
    pass
layout["How_to_use_your_BAH_to_qu"] = [66, 2067, 719, 2131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/33_text_How_to_get_your_offer_accepted.png
try:
    _c33 = get_crop(33, 75, 72)
    canvas.paste(_c33, (249, 2588), _c33)
except Exception:
    pass
layout["How_to_get_your_offer_acc"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/34_text_VA.png
try:
    _c34 = get_crop(34, 66, 49)
    canvas.paste(_c34, (118, 2454), _c34)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_11_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-13/35_text_Homebuyer_Webinar.png
try:
    _c35 = get_crop(35, 75, 72)
    canvas.paste(_c35, (249, 2588), _c35)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]
