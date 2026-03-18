# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_09
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11.png
# step_index: 9/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas using provided `canvas` and `draw`
# Available fonts: font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (189, 189, 189)      # light gray status bar
hero_bg = (244, 245, 247)               # very light neutral for hero image background
card_bg = (246, 246, 250)               # subtle off-white/pale-lavender for cards
card_border = (232, 229, 243)           # faint border color for cards
divider = (236, 236, 239)               # section divider lines
ticket_border = (46, 93, 246)           # bright blue border for ticket card
shadow_color = (225, 225, 228)          # subtle shadow

w, h = canvas.size

# Top status bar
draw.rectangle([0, 0, w, 80], fill=status_bar_color)

# Hero image background area (behind the hero image that will be pasted)
hero_margin_lr = 40
hero_top = 80
hero_bottom = 440
hero_radius = 14
draw.rounded_rectangle(
    [hero_margin_lr, hero_top, w - hero_margin_lr, hero_bottom],
    radius=hero_radius,
    fill=hero_bg,
    outline=None
)

# Subtle bottom divider under hero area
draw.line([hero_margin_lr, hero_bottom + 16, w - hero_margin_lr, hero_bottom + 16], fill=divider, width=1)

# Organizer / follow card background (rounded card behind avatar, organizer name, follow button)
org_card_top = 1180
org_card_bottom = 1360
org_card_lr = 40
org_card_radius = 22
draw.rounded_rectangle(
    [org_card_lr, org_card_top, w - org_card_lr, org_card_bottom],
    radius=org_card_radius,
    fill=card_bg,
    outline=card_border,
    width=2
)

# Thin divider below organizer area (separates details from "About this event")
about_div_y = 1560
draw.line([48, about_div_y, w - 48, about_div_y], fill=divider, width=1)

# Small horizontal section separators for event metadata list (subtle lines)
meta_start_y = 1480
for i in range(3):
    y = meta_start_y + i * 80
    draw.line([72, y, w - 72, y], fill=(250, 250, 251) if i % 2 == 0 else divider, width=1)

# "About this event" divider area (thin line above about section)
about_top_line = 2048
draw.line([40, about_top_line, w - 40, about_top_line], fill=divider, width=1)

# Ticket selection card (outlined rounded rectangle)
ticket_card_top = 2230
ticket_card_bottom = 2520
ticket_card_lr = 40
ticket_card_radius = 18
# subtle shadow under ticket card
draw.rectangle([ticket_card_lr + 6, ticket_card_top + 12, w - ticket_card_lr + 6, ticket_card_bottom + 12], fill=shadow_color)
# ticket card body & border
draw.rounded_rectangle(
    [ticket_card_lr, ticket_card_top, w - ticket_card_lr, ticket_card_bottom],
    radius=ticket_card_radius,
    fill=(255, 255, 255),
    outline=ticket_border,
    width=6
)

# Small internal divider inside ticket card (to suggest separation between label and quantity area)
inner_div_y = ticket_card_top + 110
draw.line([ticket_card_lr + 24, inner_div_y, w - ticket_card_lr - 24, inner_div_y], fill=(245, 245, 247), width=1)

# Large subtle divider above the reserve button area (so the button will appear separated)
reserve_area_top = 2688
draw.line([40, reserve_area_top, w - 40, reserve_area_top], fill=divider, width=1)

# Bottom area: subtle page bottom padding shadow
draw.rectangle([0, h - 160, w, h], fill=(250, 250, 251))

# Top toolbar background band behind navigation icons (do not draw icons themselves)
toolbar_top = 100
toolbar_bottom = 180
draw.rectangle([0, toolbar_top, w, toolbar_bottom], fill=(255, 255, 255))

# Subtle vertical rule at left content margin for visual balance
draw.line([40, hero_bottom + 40, 40, ticket_card_top - 40], fill=(250, 250, 251), width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1290), _c0)
except Exception:
    pass
layout["Following"] = [946, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/02_icon_Increase.png
try:
    _c2 = get_crop(2, 96, 96)
    canvas.paste(_c2, (1224, 2444), _c2)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1296, 132)
    canvas.paste(_c3, (72, 2756), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 93, 103)
    canvas.paste(_c4, (1108, 2441), _c4)
except Exception:
    pass
layout["icon_4"] = [1108, 2441, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/05_icon_9.16.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["9.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/06_icon_Home_Lifestyle.png
try:
    _c6 = get_crop(6, 75, 72)
    canvas.paste(_c6, (249, 2588), _c6)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/07_icon_LIVE_WEBINAR.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["LIVE_WEBINAR"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/08_icon_LIVE_WEBINAR.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1116, 108), _c8)
except Exception:
    pass
layout["LIVE_WEBINAR"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 99, 64)
    canvas.paste(_c9, (1215, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1215, 0, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/10_icon_Ticket_sales_end_soon.png
try:
    _c10 = get_crop(10, 548, 84)
    canvas.paste(_c10, (39, 753), _c10)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [39, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 57, 65)
    canvas.paste(_c11, (1317, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1317, 0, 1374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/12_icon_9.16.png
try:
    _c12 = get_crop(12, 54, 61)
    canvas.paste(_c12, (182, 2), _c12)
except Exception:
    pass
layout["9.16"] = [182, 2, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 59)
    canvas.paste(_c13, (315, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [315, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 60)
    canvas.paste(_c14, (248, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [248, 3, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/15_icon_Free.png
try:
    _c15 = get_crop(15, 133, 102)
    canvas.paste(_c15, (101, 2577), _c15)
except Exception:
    pass
layout["Free"] = [101, 2577, 234, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/16_icon_ACTIVE_MILITARY_VETERANS.png
try:
    _c16 = get_crop(16, 49, 58)
    canvas.paste(_c16, (384, 5), _c16)
except Exception:
    pass
layout["ACTIVE_MILITARY_&_VETERAN"] = [384, 5, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/17_text_9.16.png
try:
    _c17 = get_crop(17, 94, 43)
    canvas.paste(_c17, (20, 17), _c17)
except Exception:
    pass
layout["9.16"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/18_text_ACTIVE_MILITARY_VETERANS.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (36, 108), _c18)
except Exception:
    pass
layout["ACTIVE_MILITARY_&_VETERAN"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/19_text_How_to_qualify_for_your_VA_loan.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["How_to_qualify_for_your_V"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/20_text_How_to_buy_a_home_with_ZERO_down.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (36, 108), _c20)
except Exception:
    pass
layout["How_to_buy_a_home_with_ZE"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/21_text_LIVE_WEBINAR.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["LIVE_WEBINAR"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/22_text_How_to_become_a_homeowner.png
try:
    _c22 = get_crop(22, 464, 50)
    canvas.paste(_c22, (185, 409), _c22)
except Exception:
    pass
layout["How_to_become_a_homeowner"] = [185, 409, 649, 459]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/23_text_WEDNESDAY_6_PM.png
try:
    _c23 = get_crop(23, 297, 50)
    canvas.paste(_c23, (923, 414), _c23)
except Exception:
    pass
layout["WEDNESDAY_6_PM"] = [923, 414, 1220, 464]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/24_text_HEROES.png
try:
    _c24 = get_crop(24, 214, 52)
    canvas.paste(_c24, (187, 541), _c24)
except Exception:
    pass
layout["HEROES"] = [187, 541, 401, 593]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/25_text_HEROES.png
try:
    _c25 = get_crop(25, 217, 56)
    canvas.paste(_c25, (432, 539), _c25)
except Exception:
    pass
layout["HEROES"] = [432, 539, 649, 595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/26_text_Wednesday_March_20.png
try:
    _c26 = get_crop(26, 557, 73)
    canvas.paste(_c26, (43, 886), _c26)
except Exception:
    pass
layout["Wednesday;_March_20"] = [43, 886, 600, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/27_text_6.00_PM.png
try:
    _c27 = get_crop(27, 209, 54)
    canvas.paste(_c27, (625, 893), _c27)
except Exception:
    pass
layout["6.00_PM"] = [625, 893, 834, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/28_text_Active_Military_Veterans_VA.png
try:
    _c28 = get_crop(28, 475, 144)
    canvas.paste(_c28, (288, 1250), _c28)
except Exception:
    pass
layout["Active_Military_&_Veteran"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/29_text_Homebuyer_Webinar.png
try:
    _c29 = get_crop(29, 475, 144)
    canvas.paste(_c29, (288, 1250), _c29)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/30_text_HeroesSanDiego.com.png
try:
    _c30 = get_crop(30, 475, 144)
    canvas.paste(_c30, (288, 1250), _c30)
except Exception:
    pass
layout["HeroesSanDiego.com"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/31_text_EROI.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (96, 1289), _c31)
except Exception:
    pass
layout["EROI"] = [96, 1289, 240, 1433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/32_text_316_Followers.png
try:
    _c32 = get_crop(32, 475, 144)
    canvas.paste(_c32, (288, 1250), _c32)
except Exception:
    pass
layout["316_Followers"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/33_text_Online_event.png
try:
    _c33 = get_crop(33, 274, 55)
    canvas.paste(_c33, (139, 1563), _c33)
except Exception:
    pass
layout["Online_event"] = [139, 1563, 413, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/34_text_hrs_30_mins.png
try:
    _c34 = get_crop(34, 255, 54)
    canvas.paste(_c34, (176, 1672), _c34)
except Exception:
    pass
layout["hrs_30_mins"] = [176, 1672, 431, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/35_text_Refund_policy.png
try:
    _c35 = get_crop(35, 299, 63)
    canvas.paste(_c35, (138, 1780), _c35)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/36_text_The_organizer_will_review_refund_request.png
try:
    _c36 = get_crop(36, 1344, 144)
    canvas.paste(_c36, (48, 1517), _c36)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/37_text_About_this_event.png
try:
    _c37 = get_crop(37, 454, 61)
    canvas.paste(_c37, (45, 2080), _c37)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/38_text_VA.png
try:
    _c38 = get_crop(38, 66, 49)
    canvas.paste(_c38, (118, 2454), _c38)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_09_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-11/39_text_Homebuyer_Webinar.png
try:
    _c39 = get_crop(39, 75, 72)
    canvas.paste(_c39, (249, 2588), _c39)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]
