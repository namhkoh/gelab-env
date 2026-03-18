# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_08
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10.png
# step_index: 8/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([(0, 0), (1440, 2960)], fill="#fbfafc")

# Status bar (top area)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#cfcfcf")

# Subtle divider under status bar to separate from content
draw.line([(0, status_h), (1440, status_h)], fill="#bfbfbf", width=1)

# Hero/banner placeholder (rounded rectangle)
hero_margin_x = 40
hero_top = status_h + 20
hero_bottom = hero_top + 440
draw.rounded_rectangle(
    [(hero_margin_x, hero_top), (1440 - hero_margin_x, hero_bottom)],
    radius=28,
    fill="#e8e8ec",
    outline="#d6d6db",
    width=1
)

# Soft shadow under hero to lift it from background
shadow_top = hero_bottom
for i, alpha_shade in enumerate([220, 200, 180, 150]):
    y0 = shadow_top + i * 3
    y1 = y0 + 3
    shade = "#%02x%02x%02x" % (230 - i*6, 230 - i*6, 230 - i*6)
    draw.rectangle([(hero_margin_x + 6, y0), (1440 - hero_margin_x - 6, y1)], fill=shade)

# Main white content card area underneath hero (rounded top corners)
content_top = hero_bottom + 20
content_bottom = 2320  # leave room at bottom for the reserve section which will be pasted later
draw.rounded_rectangle(
    [(0, content_top), (1440, content_bottom)],
    radius=36,
    fill="#ffffff",
    outline=None
)

# Thin subtle vertical padding markers (not UI elements, just gentle separators)
# Left content margin guide (visual only)
left_pad = 48
right_pad = 1392

# Organizer / profile card background (rounded rect)
org_card_top = 1180
org_card_bottom = 1370
org_card_left = left_pad
org_card_right = right_pad
draw.rounded_rectangle(
    [(org_card_left, org_card_top), (org_card_right, org_card_bottom)],
    radius=28,
    fill="#f5f4f7",
    outline="#e9e8eb",
    width=1
)

# Small divider lines between informational rows
divider1_y = 1510
draw.line([(left_pad, divider1_y), (right_pad, divider1_y)], fill="#efeef2", width=2)

divider2_y = 1760
draw.line([(left_pad, divider2_y), (right_pad, divider2_y)], fill="#efeef2", width=1)

# "About this event" section separator area (just a subtle rule)
about_sep_y = 2028
draw.line([(left_pad, about_sep_y), (right_pad, about_sep_y)], fill="#efedf2", width=2)

# Ticket selection card (sub-section above the reserve button area)
ticket_card_top = 2140
ticket_card_bottom = 2288
ticket_card_left = 48
ticket_card_right = 1392
draw.rounded_rectangle(
    [(ticket_card_left, ticket_card_top), (ticket_card_right, ticket_card_bottom)],
    radius=18,
    fill="#ffffff",
    outline="#3f57d6",  # subtle blue outline like a selectable card
    width=6
)

# Inner subtle background inside ticket card to suggest grouped content
inner_margin = 18
draw.rounded_rectangle(
    [(ticket_card_left + inner_margin, ticket_card_top + inner_margin),
     (ticket_card_right - inner_margin, ticket_card_bottom - inner_margin)],
    radius=12,
    fill="#ffffff",
    outline="#e9e9ef",
    width=1
)

# Final subtle horizontal divider just above the reserved-area (so pasted button sits cleanly)
reserve_top = 2324  # reserved area where the big "Reserve a spot" will be pasted
draw.line([(0, reserve_top - 16), (1440, reserve_top - 16)], fill="#efeef2", width=2)

# Small decorative bottom shadow under content area to separate from reserve area
for i in range(6):
    shade = 240 - i * 6
    shade_col = "#%02x%02x%02x" % (shade, shade, shade)
    y0 = reserve_top - 12 + i
    draw.line([(left_pad, y0), (right_pad, y0)], fill=shade_col, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 111, 105)
    canvas.paste(_c1, (988, 2440), _c1)
except Exception:
    pass
layout["icon_1"] = [988, 2440, 1099, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1440, 636)
    canvas.paste(_c2, (0, 2324), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 105, 103)
    canvas.paste(_c3, (1218, 2441), _c3)
except Exception:
    pass
layout["icon_3"] = [1218, 2441, 1323, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 93, 103)
    canvas.paste(_c4, (1108, 2441), _c4)
except Exception:
    pass
layout["icon_4"] = [1108, 2441, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/05_icon_9.16.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["9.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/06_icon_Home_Lifestyle.png
try:
    _c6 = get_crop(6, 1440, 636)
    canvas.paste(_c6, (0, 2324), _c6)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/07_icon_LIVE_WEBINAR.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["LIVE_WEBINAR"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/08_icon_LIVE_WEBINAR.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1116, 108), _c8)
except Exception:
    pass
layout["LIVE_WEBINAR"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/09_icon_HeroesSanDiego.com.png
try:
    _c9 = get_crop(9, 475, 144)
    canvas.paste(_c9, (288, 1250), _c9)
except Exception:
    pass
layout["HeroesSanDiego.com"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 99, 62)
    canvas.paste(_c10, (1215, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1215, 1, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/11_icon_Ticket_sales_end_soon.png
try:
    _c11 = get_crop(11, 548, 84)
    canvas.paste(_c11, (39, 753), _c11)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [39, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 65)
    canvas.paste(_c12, (1317, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1317, 0, 1374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/13_icon_9.16.png
try:
    _c13 = get_crop(13, 54, 61)
    canvas.paste(_c13, (182, 2), _c13)
except Exception:
    pass
layout["9.16"] = [182, 2, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/14_icon_Free.png
try:
    _c14 = get_crop(14, 134, 103)
    canvas.paste(_c14, (100, 2576), _c14)
except Exception:
    pass
layout["Free"] = [100, 2576, 234, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 59)
    canvas.paste(_c15, (315, 4), _c15)
except Exception:
    pass
layout["icon_15"] = [315, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 59)
    canvas.paste(_c16, (248, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [248, 3, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/17_icon_ACTIVE_MILITARY_VETERANS.png
try:
    _c17 = get_crop(17, 49, 58)
    canvas.paste(_c17, (384, 5), _c17)
except Exception:
    pass
layout["ACTIVE_MILITARY_&_VETERAN"] = [384, 5, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/18_icon_Free.png
try:
    _c18 = get_crop(18, 103, 113)
    canvas.paste(_c18, (233, 2574), _c18)
except Exception:
    pass
layout["Free"] = [233, 2574, 336, 2687]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/19_text_9.16.png
try:
    _c19 = get_crop(19, 94, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.16"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/20_text_ACTIVE_MILITARY_VETERANS.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (36, 108), _c20)
except Exception:
    pass
layout["ACTIVE_MILITARY_&_VETERAN"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/21_text_How_to_qualify_for_your_VA_loan.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (36, 108), _c21)
except Exception:
    pass
layout["How_to_qualify_for_your_V"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/22_text_How_to_buy_a_home_with_ZERO_down.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (36, 108), _c22)
except Exception:
    pass
layout["How_to_buy_a_home_with_ZE"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/23_text_LIVE_WEBINAR.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1116, 108), _c23)
except Exception:
    pass
layout["LIVE_WEBINAR"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/24_text_How_to_become_a_homeowner.png
try:
    _c24 = get_crop(24, 464, 50)
    canvas.paste(_c24, (185, 409), _c24)
except Exception:
    pass
layout["How_to_become_a_homeowner"] = [185, 409, 649, 459]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/25_text_WEDNESDAY_6_PM.png
try:
    _c25 = get_crop(25, 297, 50)
    canvas.paste(_c25, (923, 414), _c25)
except Exception:
    pass
layout["WEDNESDAY_6_PM"] = [923, 414, 1220, 464]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/26_text_HEROES.png
try:
    _c26 = get_crop(26, 214, 52)
    canvas.paste(_c26, (187, 541), _c26)
except Exception:
    pass
layout["HEROES"] = [187, 541, 401, 593]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/27_text_HEROES.png
try:
    _c27 = get_crop(27, 217, 56)
    canvas.paste(_c27, (432, 539), _c27)
except Exception:
    pass
layout["HEROES"] = [432, 539, 649, 595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/28_text_Wednesday_March_20.png
try:
    _c28 = get_crop(28, 557, 73)
    canvas.paste(_c28, (43, 886), _c28)
except Exception:
    pass
layout["Wednesday;_March_20"] = [43, 886, 600, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/29_text_6.00_PM.png
try:
    _c29 = get_crop(29, 209, 54)
    canvas.paste(_c29, (625, 893), _c29)
except Exception:
    pass
layout["6.00_PM"] = [625, 893, 834, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/30_text_Active_Military_Veterans_VA.png
try:
    _c30 = get_crop(30, 475, 144)
    canvas.paste(_c30, (288, 1250), _c30)
except Exception:
    pass
layout["Active_Military_&_Veteran"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/31_text_Homebuyer_Webinar.png
try:
    _c31 = get_crop(31, 475, 144)
    canvas.paste(_c31, (288, 1250), _c31)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [288, 1250, 763, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/32_text_Online_event.png
try:
    _c32 = get_crop(32, 274, 55)
    canvas.paste(_c32, (139, 1563), _c32)
except Exception:
    pass
layout["Online_event"] = [139, 1563, 413, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/33_text_hrs_30_mins.png
try:
    _c33 = get_crop(33, 255, 54)
    canvas.paste(_c33, (176, 1672), _c33)
except Exception:
    pass
layout["hrs_30_mins"] = [176, 1672, 431, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/34_text_Refund_policy.png
try:
    _c34 = get_crop(34, 299, 63)
    canvas.paste(_c34, (138, 1780), _c34)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/35_text_The_organizer_will_review_refund_request.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 1517), _c35)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/36_text_About_this_event.png
try:
    _c36 = get_crop(36, 454, 61)
    canvas.paste(_c36, (45, 2080), _c36)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/37_text_VA.png
try:
    _c37 = get_crop(37, 66, 49)
    canvas.paste(_c37, (118, 2454), _c37)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/38_text_Homebuyer_Webinar.png
try:
    _c38 = get_crop(38, 453, 80)
    canvas.paste(_c38, (186, 2440), _c38)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [186, 2440, 639, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_08_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-10/39_clickable_Organizer_profile_picture.png
try:
    _c39 = get_crop(39, 144, 144)
    canvas.paste(_c39, (96, 1289), _c39)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1289, 240, 1433]
