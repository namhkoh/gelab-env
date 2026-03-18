# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_12
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14.png
# step_index: 12/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for Event page (PIL ImageDraw provided)
# Available variables: canvas (1440x2960), draw (ImageDraw), fonts (font_sm, font_md, font_lg, font_xl)

# Canvas background (slightly warm white to match screenshot)
draw.rectangle((0, 0, 1440, 2960), fill="#fbfbfb")

# Status bar (top strip)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#d6d6d6")

# Header / toolbar area below status bar
header_y0 = status_h
header_y1 = status_h + 120
draw.rectangle((0, header_y0, 1440, header_y1), fill="#ffffff")
# Header bottom divider
draw.line((24, header_y1, 1416, header_y1), fill="#e6e6e6", width=1)

# Subtle shadowed container behind main content (large rounded card impression)
main_shadow = (28, 188, 1412, 1628)
draw.rounded_rectangle(main_shadow, radius=28, fill="#efefef")  # shadow/backing
main_card = (36, 180, 1404, 1620)
draw.rounded_rectangle(main_card, radius=24, fill="#ffffff")

# Divider separating intro text area and rest of content
divider_y = 1700
draw.line((36, divider_y, 1404, divider_y), fill="#f0f0f0", width=1)

# Image / media area background (rounded rectangle with subtle fill)
img_y0 = 1780
img_y1 = 2240
img_rect = (36, img_y0, 1404, img_y1)
draw.rounded_rectangle(img_rect, radius=28, fill="#f3f4f6")

# Left and right soft vignette bands to mimic blurred edges in screenshot
# (Keep very subtle so pasted image will overlay naturally)
draw.rectangle((0, img_y0, 120, img_y1), fill="#eef0ef")
draw.rectangle((1320, img_y0, 1440, img_y1), fill="#eef0ef")

# Light separator above ticket/card area
sep_y = 2320
draw.line((36, sep_y, 1404, sep_y), fill="#eeeeee", width=1)

# Ticket / selection card (white with purple outline, rounded)
ticket_y0 = 2360
ticket_y1 = 2660
ticket_rect = (36, ticket_y0, 1404, ticket_y1)
# shadow backing for the ticket card
draw.rounded_rectangle((40, ticket_y0+6, 1408, ticket_y1+10), radius=20, fill="#f2f3f7")
# main ticket card
draw.rounded_rectangle(ticket_rect, radius=20, fill="#ffffff", outline="#4a2b6f", width=6)

# Inner horizontal divider inside ticket card (subtle)
inner_div_y = ticket_y0 + 70
draw.line((60, inner_div_y, 1380, inner_div_y), fill="#f0f0f3", width=1)

# Small pill background for quantity area (background only, icons/text to be pasted on top)
qty_box = (1120, ticket_y0 + 24, 1376, ticket_y0 + 104)
draw.rounded_rectangle(qty_box, radius=12, fill="#f7f7fb")

# Add subtle bottom area band (to separate from Reserve button area)
bottom_band_y = 2720
draw.rectangle((0, bottom_band_y, 1440, 2960), fill="#ffffff")
draw.line((36, bottom_band_y, 1404, bottom_band_y), fill="#f0f0f0", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/02_icon_Decrease.png
try:
    _c2 = get_crop(2, 99, 96)
    canvas.paste(_c2, (996, 2444), _c2)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/03_icon_9.16.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/04_icon_Increase.png
try:
    _c4 = get_crop(4, 96, 96)
    canvas.paste(_c4, (1224, 2444), _c4)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 93, 104)
    canvas.paste(_c5, (1108, 2440), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2440, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 56)
    canvas.paste(_c6, (316, 7), _c6)
except Exception:
    pass
layout["icon_6"] = [316, 7, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/07_icon_Reserve_a_spot.png
try:
    _c7 = get_crop(7, 1296, 132)
    canvas.paste(_c7, (72, 2756), _c7)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 55)
    canvas.paste(_c8, (249, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [249, 6, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 55)
    canvas.paste(_c9, (183, 6), _c9)
except Exception:
    pass
layout["icon_9"] = [183, 6, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/10_icon_Free.png
try:
    _c10 = get_crop(10, 132, 111)
    canvas.paste(_c10, (101, 2571), _c10)
except Exception:
    pass
layout["Free"] = [101, 2571, 233, 2682]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 44, 56)
    canvas.paste(_c11, (1326, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [1326, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 89, 57)
    canvas.paste(_c12, (1217, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1217, 3, 1306, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/13_icon_9.16.png
try:
    _c13 = get_crop(13, 51, 58)
    canvas.paste(_c13, (118, 4), _c13)
except Exception:
    pass
layout["9.16"] = [118, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/14_icon_Free.png
try:
    _c14 = get_crop(14, 75, 72)
    canvas.paste(_c14, (249, 2588), _c14)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/15_icon_SOLDI.png
try:
    _c15 = get_crop(15, 99, 96)
    canvas.paste(_c15, (996, 2444), _c15)
except Exception:
    pass
layout["SOLDI"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 46, 61)
    canvas.paste(_c16, (385, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [385, 3, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/17_icon_Active_Military_Ve.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (36, 108), _c17)
except Exception:
    pass
layout["Active_Military_&_Ve_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/18_text_9.16.png
try:
    _c18 = get_crop(18, 91, 43)
    canvas.paste(_c18, (20, 17), _c18)
except Exception:
    pass
layout["9.16"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/19_text_About_our_Heroes_Nationwide_Relocation_S.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["About_our_Heroes_Nationwi"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/20_text_This_is_a_FREE_Webinar_sponsored_by_Hero.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1116, 108), _c20)
except Exception:
    pass
layout["This_is_a_FREE_Webinar_sp"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/21_text_Ready_to_take_the_next_step_Register_for.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["Ready_to_take_the_next_st"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/22_text_out_on_this_opportunity_to_gain_valuable.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1116, 108), _c22)
except Exception:
    pass
layout["out_on_this_opportunity_t"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/23_text_Get_your_questions_answered_this_Wednesd.png
try:
    _c23 = get_crop(23, 1256, 73)
    canvas.paste(_c23, (40, 944), _c23)
except Exception:
    pass
layout["Get_your_questions_answer"] = [40, 944, 1296, 1017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/24_text_comfort_of_your_home.png
try:
    _c24 = get_crop(24, 468, 61)
    canvas.paste(_c24, (41, 1017), _c24)
except Exception:
    pass
layout["comfort_of_your_home:"] = [41, 1017, 509, 1078]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/25_text_elem_25.png
try:
    _c25 = get_crop(25, 57, 25)
    canvas.paste(_c25, (44, 1147), _c25)
except Exception:
    pass
layout["#++"] = [44, 1147, 101, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/26_text_elem_26.png
try:
    _c26 = get_crop(26, 55, 25)
    canvas.paste(_c26, (821, 1209), _c26)
except Exception:
    pass
layout["***"] = [821, 1209, 876, 1234]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/27_text_Register_now_and_secure_your_spot_today.png
try:
    _c27 = get_crop(27, 873, 67)
    canvas.paste(_c27, (111, 1330), _c27)
except Exception:
    pass
layout["Register_now_and_secure_y"] = [111, 1330, 984, 1397]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/28_text_MilitaryHomebuyers_VeteransAtHome_VAHome.png
try:
    _c28 = get_crop(28, 1198, 61)
    canvas.paste(_c28, (41, 1521), _c28)
except Exception:
    pass
layout["#MilitaryHomebuyers_#Vete"] = [41, 1521, 1239, 1582]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/29_text_HomeownershipForHeroes_RegionRealEstate.png
try:
    _c29 = get_crop(29, 1023, 63)
    canvas.paste(_c29, (41, 1583), _c29)
except Exception:
    pass
layout["#HomeownershipForHeroes_#"] = [41, 1583, 1064, 1646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/30_text_heroes_heroessandiego_VAloan_military_sa.png
try:
    _c30 = get_crop(30, 99, 96)
    canvas.paste(_c30, (996, 2444), _c30)
except Exception:
    pass
layout["#heroes_#heroessandiego_#"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/31_text_home.png
try:
    _c31 = get_crop(31, 165, 56)
    canvas.paste(_c31, (41, 1837), _c31)
except Exception:
    pass
layout["#home"] = [41, 1837, 206, 1893]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/32_text_VA.png
try:
    _c32 = get_crop(32, 66, 49)
    canvas.paste(_c32, (118, 2454), _c32)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_12_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-14/33_text_Homebuyer_Webinar.png
try:
    _c33 = get_crop(33, 75, 72)
    canvas.paste(_c33, (249, 2588), _c33)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]
