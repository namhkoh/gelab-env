# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_13
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15.png
# step_index: 13/15
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
draw.rectangle([(0, 0), (1440, 72)], fill="#d0d0d0")

# Header / toolbar area (below status bar)
draw.rectangle([(0, 72), (1440, 176)], fill="#ffffff")
# thin divider under header
draw.line([(28, 176), (1412, 176)], fill="#e6e6e6", width=2)

# Large media container backgrounds (behind the two stacked video/image areas)
# First media container (around pos y ~127..882)
media1_outer = (48, 112, 1392, 904)
draw.rounded_rectangle(media1_outer, radius=12, fill="#fbfbfb", outline="#e6e6e9")
# subtle inner shadow for depth
draw.rectangle([(media1_outer[0]+6, media1_outer[1]+6), (media1_outer[2]-6, media1_outer[3]-6)], outline="#f0f0f1")

# Second media container (stacked below first, around y ~903..1658)
media2_outer = (48, 880, 1392, 1640)
draw.rounded_rectangle(media2_outer, radius=12, fill="#fbfbfb", outline="#e6e6e9")
draw.rectangle([(media2_outer[0]+6, media2_outer[1]+6), (media2_outer[2]-6, media2_outer[3]-6)], outline="#f0f0f1")

# Separator line below media area
sep_y = 1680
draw.line([(28, sep_y), (1412, sep_y)], fill="#efeff2", width=2)

# Agenda heading area (background spacing)
# keep it minimal (no text), just a clear white band for the heading region
draw.rectangle([(0, 1740), (1440, 1900)], fill="#ffffff")
draw.line([(28, 1900), (1412, 1900)], fill="#f0f0f3", width=1)

# Agenda card - light, warm rounded card for the agenda item
agenda_card = (48, 1960, 1392, 2380)
draw.rounded_rectangle(agenda_card, radius=18, fill="#fff5f4", outline="#f0dcd9")
# left accent bar inside the agenda card
accent_x = agenda_card[0] + 18
draw.rectangle([(accent_x, agenda_card[1]+24), (accent_x+8, agenda_card[3]-24)], fill="#f4b5a6")

# subtle inner content area for the agenda (no text)
draw.rectangle([(agenda_card[0]+20, agenda_card[1]+20), (agenda_card[2]-20, agenda_card[3]-20)], outline="#f6eaea")

# Divider above ticket/reservation panel
draw.line([(28, 2388), (1412, 2388)], fill="#f0f0f3", width=1)

# Ticket/reservation panel (rounded white panel with blue-ish outline)
panel_outer = (36, 2360, 1404, 2728)
# outer border emulation
draw.rounded_rectangle(panel_outer, radius=22, fill="#ffffff", outline="#3d3fcf")
# inner fill (slightly inset to show thick border)
inner = (panel_outer[0]+8, panel_outer[1]+8, panel_outer[2]-8, panel_outer[3]-8)
draw.rounded_rectangle(inner, radius=16, fill="#ffffff", outline=None)
# subtle shadow under panel
shadow = (panel_outer[0]+6, panel_outer[3]+6, panel_outer[2]-6, panel_outer[3]+18)
draw.rectangle(shadow, fill="#000000", outline=None)
# make shadow very faint by blending (draw a few translucent rectangles)
# (Use progressively lighter fills to mimic blur)
draw.rectangle([(panel_outer[0]+6, panel_outer[3]+6), (panel_outer[2]-6, panel_outer[3]+10)], fill="#080808")
draw.rectangle([(panel_outer[0]+8, panel_outer[3]+10), (panel_outer[2]-8, panel_outer[3]+14)], fill="#101010")

# thin top divider for the reserve button area (keeps separation)
draw.line([(28, 2748), (1412, 2748)], fill="#efe9e5", width=1)

# Page background subtle tint for the lower region to separate the reserve action area
draw.rectangle([(0, 2680), (1440, 2960)], fill="#ffffff")

# Small decorative bottom safe-area padding bar (no interactive elements)
draw.rectangle([(0, 2920), (1440, 2960)], fill="#fafafa")

# Add a few faint vertical guides for content alignment (very subtle)
for x in (48, 360, 720, 1080, 1392):
    draw.line([(x, 176), (x, 2600)], fill="#fbfbfc", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/02_icon_Acom_herpessandiego.png
try:
    _c2 = get_crop(2, 1323, 755)
    canvas.paste(_c2, (58, 903), _c2)
except Exception:
    pass
layout["Acom_herpessandiego"] = [58, 903, 1381, 1658]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1296, 132)
    canvas.paste(_c3, (72, 2756), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2444), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 94, 103)
    canvas.paste(_c6, (1108, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2441, 1202, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 58)
    canvas.paste(_c7, (316, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [316, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 54, 58)
    canvas.paste(_c8, (247, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 4, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/09_icon_9.16.png
try:
    _c9 = get_crop(9, 54, 59)
    canvas.paste(_c9, (182, 2), _c9)
except Exception:
    pass
layout["9.16"] = [182, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/10_icon_613.png
try:
    _c10 = get_crop(10, 1323, 755)
    canvas.paste(_c10, (58, 127), _c10)
except Exception:
    pass
layout["613"] = [58, 127, 1381, 882]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 50, 52)
    canvas.paste(_c11, (1320, 8), _c11)
except Exception:
    pass
layout["icon_11"] = [1320, 8, 1370, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/12_icon_9.16.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (36, 108), _c12)
except Exception:
    pass
layout["9.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 86, 52)
    canvas.paste(_c13, (1227, 8), _c13)
except Exception:
    pass
layout["icon_13"] = [1227, 8, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/14_icon_Free.png
try:
    _c14 = get_crop(14, 137, 110)
    canvas.paste(_c14, (98, 2569), _c14)
except Exception:
    pass
layout["Free"] = [98, 2569, 235, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 48, 60)
    canvas.paste(_c15, (383, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [383, 3, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/16_icon_9.16.png
try:
    _c16 = get_crop(16, 51, 60)
    canvas.paste(_c16, (118, 3), _c16)
except Exception:
    pass
layout["9.16"] = [118, 3, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/17_icon_Share.png
try:
    _c17 = get_crop(17, 65, 84)
    canvas.paste(_c17, (1285, 903), _c17)
except Exception:
    pass
layout["Share"] = [1285, 903, 1350, 987]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/18_icon_Free.png
try:
    _c18 = get_crop(18, 75, 72)
    canvas.paste(_c18, (249, 2588), _c18)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/19_icon_Active_Military_Ve_-.png
try:
    _c19 = get_crop(19, 1323, 755)
    canvas.paste(_c19, (58, 127), _c19)
except Exception:
    pass
layout["Active_Military_&_Ve_-"] = [58, 127, 1381, 882]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/20_text_9.16.png
try:
    _c20 = get_crop(20, 91, 43)
    canvas.paste(_c20, (20, 17), _c20)
except Exception:
    pass
layout["9.16"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/21_text_Read_less.png
try:
    _c21 = get_crop(21, 206, 144)
    canvas.paste(_c21, (48, 1677), _c21)
except Exception:
    pass
layout["Read_less"] = [48, 1677, 254, 1821]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/22_text_Agenda.png
try:
    _c22 = get_crop(22, 231, 77)
    canvas.paste(_c22, (41, 1938), _c22)
except Exception:
    pass
layout["Agenda"] = [41, 1938, 272, 2015]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/23_text_6.00_PM.png
try:
    _c23 = get_crop(23, 165, 45)
    canvas.paste(_c23, (219, 2210), _c23)
except Exception:
    pass
layout["6.00_PM"] = [219, 2210, 384, 2255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/24_text_Welcome_Real_Estate_Market_Undate.png
try:
    _c24 = get_crop(24, 99, 96)
    canvas.paste(_c24, (996, 2444), _c24)
except Exception:
    pass
layout["Welcome_&_Real_Estate_Mar"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/25_text_VA.png
try:
    _c25 = get_crop(25, 66, 49)
    canvas.paste(_c25, (118, 2454), _c25)
except Exception:
    pass
layout["VA"] = [118, 2454, 184, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/26_text_Homebuyer_Webinar.png
try:
    _c26 = get_crop(26, 75, 72)
    canvas.paste(_c26, (249, 2588), _c26)
except Exception:
    pass
layout["Homebuyer_Webinar"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/27_clickable_Play.png
try:
    _c27 = get_crop(27, 93, 66)
    canvas.paste(_c27, (673, 472), _c27)
except Exception:
    pass
layout["Play"] = [673, 472, 766, 538]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/28_clickable_Watch_on_YouTube.png
try:
    _c28 = get_crop(28, 238, 65)
    canvas.paste(_c28, (58, 810), _c28)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 810, 296, 875]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/29_clickable_Photo_image_of_Heroes_San_Diego.png
try:
    _c29 = get_crop(29, 66, 66)
    canvas.paste(_c29, (68, 913), _c29)
except Exception:
    pass
layout["Photo_image_of_Heroes_San"] = [68, 913, 134, 979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/30_clickable_Heroes_San_Diego_-_Helping_Heroes_Achiev.png
try:
    _c30 = get_crop(30, 1127, 33)
    canvas.paste(_c30, (144, 932), _c30)
except Exception:
    pass
layout["Heroes_San_Diego_-_Helpin"] = [144, 932, 1271, 965]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/31_clickable_Play.png
try:
    _c31 = get_crop(31, 93, 66)
    canvas.paste(_c31, (673, 1248), _c31)
except Exception:
    pass
layout["Play"] = [673, 1248, 766, 1314]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_13_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-15/32_clickable_Watch_on_YouTube.png
try:
    _c32 = get_crop(32, 238, 65)
    canvas.paste(_c32, (58, 1586), _c32)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 1586, 296, 1651]
