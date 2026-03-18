# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_12
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14.png
# step_index: 12/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background base
draw.rectangle([(0, 0), canvas.size], fill="#FAFBFD")

# Status bar (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#DADDE0")

# Header area (search/title row) beneath status bar
header_top = status_h
header_h = 110
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill="#FFFFFF")
# header bottom divider
draw.line([(32, header_top + header_h), (1440 - 32, header_top + header_h)], fill="#E6E7EA", width=2)

# Location row divider (thin)
loc_div_y = 260
draw.line([(32, loc_div_y), (1440 - 32, loc_div_y)], fill="#F0F1F3", width=1)

# Filter chips background pills (do not draw text/icons)
chips = [
    # (x, y, w, h, color)
    (54, 410, 372, 103, "#2F56F3"),   # active filter (deep blue)
    (438, 410, 400, 103, "#E8F2FF"),  # Anytime (light blue)
    (850, 410, 187, 103, "#E8F2FF"),  # Music
    (1049, 410, 241, 103, "#E8F2FF"), # Business
    (1295, 406, 139, 111, "#E8F2FF"), # trailing chip
]
for x, y, w, h, col in chips:
    # Slightly inset to create soft pill look
    bbox = [x+4, y+6, x + w - 4, y + h - 6]
    radius = int((bbox[3] - bbox[1]) / 2)
    draw.rounded_rectangle(bbox, radius=radius, fill=col, outline=None)

# Large content card containers (do not draw inner text/images)
# First event block container (using detected bounding box)
card1_x, card1_y, card1_w, card1_h = 48, 676, 1344, 1175
card1_bbox = [card1_x - 8, card1_y - 8, card1_x + card1_w + 8, card1_y + card1_h + 8]
# shadow
shadow_bbox = [card1_bbox[0]+8, card1_bbox[1]+10, card1_bbox[2]+8, card1_bbox[3]+10]
draw.rounded_rectangle(shadow_bbox, radius=20, fill="#E9EBEF")
# card background
draw.rounded_rectangle(card1_bbox, radius=20, fill="#FFFFFF", outline="#E6E7EA", width=1)

# Separator between first and second event
sep_y = card1_y + card1_h + 24
draw.line([(48, sep_y), (1440 - 48, sep_y)], fill="#F1F2F4", width=1)

# Second event block container
card2_x, card2_y, card2_w, card2_h = 48, 1899, 1344, 917
card2_bbox = [card2_x - 8, card2_y - 8, card2_x + card2_w + 8, card2_y + card2_h + 8]
# shadow
shadow2_bbox = [card2_bbox[0]+6, card2_bbox[1]+8, card2_bbox[2]+6, card2_bbox[3]+8]
draw.rounded_rectangle(shadow2_bbox, radius=20, fill="#EDEFF1")
# card background
draw.rounded_rectangle(card2_bbox, radius=20, fill="#FFFFFF", outline="#E6E7EA", width=1)

# Thin divider lines between content groups (safe structural elements)
# Under filter area
draw.line([(32, 480), (1440 - 32, 480)], fill="#ECEEF1", width=1)
# Between cards and lower content
draw.line([(32, card2_y - 36), (1440 - 32, card2_y - 36)], fill="#F2F3F5", width=1)

# Bottom navigation bar background
nav_h = 116
nav_top = canvas.size[1] - nav_h
draw.rectangle([(0, nav_top), (1440, canvas.size[1])], fill="#FFFFFF")
# nav top divider
draw.line([(24, nav_top), (1440 - 24, nav_top)], fill="#E8EAED", width=2)

# Small highlight bar under header search area (thin)
draw.line([(32, header_top + header_h + 6), (1440 - 32, header_top + header_h + 6)], fill="#F4F6F8", width=1)

# Gentle left margin vertical guide (visual structure only)
draw.line([(48, header_top), (48, canvas.size[1] - nav_h)], fill="#F7F8FA", width=1)

# Gentle right margin vertical guide (visual structure only)
draw.line([(1440 - 48, header_top), (1440 - 48, canvas.size[1] - nav_h)], fill="#F7F8FA", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1049, 410), _c1)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (438, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/04_icon_Fo.png
try:
    _c4 = get_crop(4, 139, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1434, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/05_icon_VA_22046.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["VA_22046"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2415), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 66)
    canvas.paste(_c7, (1152, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1152, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/09_icon_4.45.png
try:
    _c9 = get_crop(9, 124, 116)
    canvas.paste(_c9, (55, 113), _c9)
except Exception:
    pass
layout["4.45"] = [55, 113, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/10_icon_Ho_F.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1092, 1192), _c10)
except Exception:
    pass
layout["Ho?F"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/11_icon_Wellness.png
try:
    _c11 = get_crop(11, 64, 63)
    canvas.paste(_c11, (309, 0), _c11)
except Exception:
    pass
layout["Wellness"] = [309, 0, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 97, 63)
    canvas.paste(_c12, (1212, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1212, 0, 1309, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/13_icon_entertainment_family_fun_and.png
try:
    _c13 = get_crop(13, 1344, 1175)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["entertainment,_family_fun"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/14_icon_4.45.png
try:
    _c14 = get_crop(14, 58, 63)
    canvas.paste(_c14, (181, 1), _c14)
except Exception:
    pass
layout["4.45"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 61)
    canvas.paste(_c15, (250, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [250, 1, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 55, 61)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1374, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/17_icon_4.45.png
try:
    _c17 = get_crop(17, 57, 64)
    canvas.paste(_c17, (116, 0), _c17)
except Exception:
    pass
layout["4.45"] = [116, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/18_icon_Wellness.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Wellness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/19_icon_Grand_Opening_Party_4Ever_Young_Falls.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Grand_Opening_Party_4Ever"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/20_icon_Wellness.png
try:
    _c20 = get_crop(20, 49, 61)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["Wellness"] = [384, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/21_icon_Promoted.png
try:
    _c21 = get_crop(21, 45, 63)
    canvas.paste(_c21, (281, 1746), _c21)
except Exception:
    pass
layout["Promoted"] = [281, 1746, 326, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/22_icon_GRAND_OPENING.png
try:
    _c22 = get_crop(22, 1344, 917)
    canvas.paste(_c22, (48, 1899), _c22)
except Exception:
    pass
layout["GRAND_OPENING"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/23_icon_Washington.png
try:
    _c23 = get_crop(23, 493, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["Washington"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/24_icon_Grand_Opening_Party_4Ever_Young_Falls.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Grand_Opening_Party_4Ever"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/25_icon_Church.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Church"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/26_icon_Ho_F.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1236, 1192), _c26)
except Exception:
    pass
layout["Ho?F"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/27_icon_4.45.png
try:
    _c27 = get_crop(27, 149, 63)
    canvas.paste(_c27, (11, 0), _c27)
except Exception:
    pass
layout["4.45"] = [11, 0, 160, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 62, 59)
    canvas.paste(_c29, (1307, 2014), _c29)
except Exception:
    pass
layout["icon_29"] = [1307, 2014, 1369, 2073]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/30_icon_Promoted.png
try:
    _c30 = get_crop(30, 243, 66)
    canvas.paste(_c30, (84, 1743), _c30)
except Exception:
    pass
layout["Promoted"] = [84, 1743, 327, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/31_icon_VA_22046.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (864, 2804), _c31)
except Exception:
    pass
layout["VA_22046"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/32_text_808_events.png
try:
    _c32 = get_crop(32, 372, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["808_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/33_text_UM_CAPITAL_REGION_MEDICAL_CENTER.png
try:
    _c33 = get_crop(33, 400, 103)
    canvas.paste(_c33, (438, 410), _c33)
except Exception:
    pass
layout["UM_CAPITAL_REGION_MEDICAL"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/34_text_901_Harry_._Truman_Drive_N_Largo_MD_2077.png
try:
    _c34 = get_crop(34, 1344, 1175)
    canvas.paste(_c34, (48, 676), _c34)
except Exception:
    pass
layout["901_Harry_$._Truman_Drive"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/35_text_University_of_Maryland_Capital_Region_Me.png
try:
    _c35 = get_crop(35, 1344, 1175)
    canvas.paste(_c35, (48, 676), _c35)
except Exception:
    pass
layout["University_of_Maryland_Ca"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/36_text_Free.png
try:
    _c36 = get_crop(36, 80, 39)
    canvas.paste(_c36, (117, 2614), _c36)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_12_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-14/37_text_Grand_Opening_Party_4Ever_Young_Falls.png
try:
    _c37 = get_crop(37, 1344, 917)
    canvas.paste(_c37, (48, 1899), _c37)
except Exception:
    pass
layout["Grand_Opening_Party_4Ever"] = [48, 1899, 1392, 2816]
