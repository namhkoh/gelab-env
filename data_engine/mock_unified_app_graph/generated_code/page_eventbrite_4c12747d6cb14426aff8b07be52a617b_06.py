# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_06
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8.png
# step_index: 6/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (ensure canvas starts with the dominant white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(230, 230, 230))  # light gray status bar

# Header / toolbar area (below status bar)
header_y0 = status_h
header_y1 = 192
draw.rectangle([(0, header_y0), (1440, header_y1)], fill=(255, 255, 255))  # header background (white)
# subtle bottom divider/shadow under header
draw.line([(24, header_y1), (1440-24, header_y1)], fill=(225, 225, 230), width=2)

# Large sections - subtle grouped backgrounds (placed behind chips & headings)
# Categories group background
cat_x0, cat_x1 = 24, 1440-24
cat_y0, cat_y1 = 340, 560
draw.rounded_rectangle([(cat_x0, cat_y0), (cat_x1, cat_y1)], radius=18, fill=(249, 250, 252), outline=None)

# Event type group background (larger block)
etype_x0, etype_x1 = 24, 1440-24
etype_y0, etype_y1 = 820, 1348
draw.rounded_rectangle([(etype_x0, etype_y0), (etype_x1, etype_y1)], radius=20, fill=(249, 250, 252), outline=None)

# Languages group background
lang_x0, lang_x1 = 24, 1440-24
lang_y0, lang_y1 = 1620, 1820
draw.rounded_rectangle([(lang_x0, lang_y0), (lang_x1, lang_y1)], radius=18, fill=(249, 250, 252), outline=None)

# Price & Only-free toggle area background (subtle)
price_x0, price_x1 = 24, 1440-24
price_y0, price_y1 = 1940, 2060
draw.rounded_rectangle([(price_x0, price_y0), (price_x1, price_y1)], radius=14, fill=(255, 255, 255), outline=(240,240,245))

# Sort by container background (behind segmented control)
sort_x0, sort_x1 = 24, 1440-24
sort_y0, sort_y1 = 2260, 2470
draw.rounded_rectangle([(sort_x0, sort_y0), (sort_x1, sort_y1)], radius=18, fill=(247, 246, 249), outline=(220,220,230))

# Top-to-bottom separators between major sections
sep_color = (235, 235, 240)
draw.line([(24, 320), (1440-24, 320)], fill=sep_color, width=1)     # under header area / before categories
draw.line([(24, 600), (1440-24, 600)], fill=sep_color, width=1)     # after categories
draw.line([(24, 800), (1440-24, 800)], fill=sep_color, width=1)     # before event types
draw.line([(24, 1368), (1440-24, 1368)], fill=sep_color, width=1)   # after event types
draw.line([(24, 1600), (1440-24, 1600)], fill=sep_color, width=1)   # before languages
draw.line([(24, 1860), (1440-24, 1860)], fill=sep_color, width=1)   # after languages
draw.line([(24, 1928), (1440-24, 1928)], fill=sep_color, width=1)   # above price block
draw.line([(24, 2160), (1440-24, 2160)], fill=sep_color, width=1)   # above sort area
draw.line([(24, 2680), (1440-24, 2680)], fill=(230,230,235), width=2)  # top of bottom action area

# Subtle shadow under some blocks to create depth
shadow_color = (240, 239, 245)
draw.rectangle([(cat_x0+6, cat_y1), (cat_x1-6, cat_y1+4)], fill=shadow_color)
draw.rectangle([(etype_x0+6, etype_y1), (etype_x1-6, etype_y1+6)], fill=shadow_color)
draw.rectangle([(sort_x0+6, sort_y1), (sort_x1-6, sort_y1+6)], fill=shadow_color)

# Decorative left/right margins: faint vertical guides (very subtle)
margin_color = (250, 250, 251)
draw.rectangle([(0, header_y1), (24, 2960)], fill=margin_color)
draw.rectangle([(1440-24, header_y1), (1440, 2960)], fill=margin_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/00_icon_Game.png
try:
    _c0 = get_crop(0, 185, 144)
    canvas.paste(_c0, (749, 1083), _c0)
except Exception:
    pass
layout["Game"] = [749, 1083, 934, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/02_icon_French.png
try:
    _c2 = get_crop(2, 205, 144)
    canvas.paste(_c2, (768, 1656), _c2)
except Exception:
    pass
layout["French"] = [768, 1656, 973, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/03_icon_Tournament.png
try:
    _c3 = get_crop(3, 302, 144)
    canvas.paste(_c3, (423, 1083), _c3)
except Exception:
    pass
layout["Tournament"] = [423, 1083, 725, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/04_icon_Community.png
try:
    _c4 = get_crop(4, 294, 144)
    canvas.paste(_c4, (848, 383), _c4)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1656), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1656, 744, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/06_icon_Race.png
try:
    _c6 = get_crop(6, 169, 144)
    canvas.paste(_c6, (958, 1083), _c6)
except Exception:
    pass
layout["Race"] = [958, 1083, 1127, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/07_icon_Class.png
try:
    _c7 = get_crop(7, 176, 144)
    canvas.paste(_c7, (856, 956), _c7)
except Exception:
    pass
layout["Class"] = [856, 956, 1032, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/08_icon_Appearance.png
try:
    _c8 = get_crop(8, 307, 144)
    canvas.paste(_c8, (563, 1210), _c8)
except Exception:
    pass
layout["Appearance"] = [563, 1210, 870, 1354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/09_icon_Expo.png
try:
    _c9 = get_crop(9, 167, 144)
    canvas.paste(_c9, (614, 829), _c9)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/10_icon_Italian.png
try:
    _c10 = get_crop(10, 191, 144)
    canvas.paste(_c10, (997, 1656), _c10)
except Exception:
    pass
layout["Italian"] = [997, 1656, 1188, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/11_icon_Screening.png
try:
    _c11 = get_crop(11, 269, 144)
    canvas.paste(_c11, (380, 956), _c11)
except Exception:
    pass
layout["Screening"] = [380, 956, 649, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/12_icon_Convention.png
try:
    _c12 = get_crop(12, 293, 144)
    canvas.paste(_c12, (805, 829), _c12)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/13_icon_Rally.png
try:
    _c13 = get_crop(13, 166, 144)
    canvas.paste(_c13, (233, 1083), _c13)
except Exception:
    pass
layout["Rally"] = [233, 1083, 399, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/14_icon_Gala.png
try:
    _c14 = get_crop(14, 159, 144)
    canvas.paste(_c14, (673, 956), _c14)
except Exception:
    pass
layout["Gala"] = [673, 956, 832, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/15_icon_Festival.png
try:
    _c15 = get_crop(15, 219, 127)
    canvas.paste(_c15, (1122, 829), _c15)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/16_icon_Party.png
try:
    _c16 = get_crop(16, 173, 127)
    canvas.paste(_c16, (36, 1083), _c16)
except Exception:
    pass
layout["Party"] = [36, 1083, 209, 1210]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/17_icon_German.png
try:
    _c17 = get_crop(17, 225, 135)
    canvas.paste(_c17, (270, 1656), _c17)
except Exception:
    pass
layout["German"] = [270, 1656, 495, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/18_icon_Networking.png
try:
    _c18 = get_crop(18, 296, 144)
    canvas.paste(_c18, (1056, 956), _c18)
except Exception:
    pass
layout["Networking"] = [1056, 956, 1352, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/19_icon_Seminar.png
try:
    _c19 = get_crop(19, 232, 144)
    canvas.paste(_c19, (358, 829), _c19)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/20_icon_Music.png
try:
    _c20 = get_crop(20, 187, 135)
    canvas.paste(_c20, (36, 383), _c20)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/21_icon_Other.png
try:
    _c21 = get_crop(21, 182, 144)
    canvas.paste(_c21, (894, 1210), _c21)
except Exception:
    pass
layout["Other"] = [894, 1210, 1076, 1354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/22_icon_Business.png
try:
    _c22 = get_crop(22, 241, 135)
    canvas.paste(_c22, (247, 383), _c22)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/23_icon_Tour.png
try:
    _c23 = get_crop(23, 156, 144)
    canvas.paste(_c23, (1151, 1083), _c23)
except Exception:
    pass
layout["Tour"] = [1151, 1083, 1307, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/24_icon_Performance.png
try:
    _c24 = get_crop(24, 320, 144)
    canvas.paste(_c24, (36, 956), _c24)
except Exception:
    pass
layout["Performance"] = [36, 956, 356, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/25_icon_Retreat.png
try:
    _c25 = get_crop(25, 215, 135)
    canvas.paste(_c25, (324, 1210), _c25)
except Exception:
    pass
layout["Retreat"] = [324, 1210, 539, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/26_icon_Arts.png
try:
    _c26 = get_crop(26, 152, 144)
    canvas.paste(_c26, (1166, 383), _c26)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/27_icon_English.png
try:
    _c27 = get_crop(27, 210, 135)
    canvas.paste(_c27, (36, 1656), _c27)
except Exception:
    pass
layout["English"] = [36, 1656, 246, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/28_icon_Conference.png
try:
    _c28 = get_crop(28, 298, 127)
    canvas.paste(_c28, (36, 829), _c28)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/29_icon_Attraction.png
try:
    _c29 = get_crop(29, 264, 135)
    canvas.paste(_c29, (36, 1210), _c29)
except Exception:
    pass
layout["Attraction"] = [36, 1210, 300, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/30_icon_Date.png
try:
    _c30 = get_crop(30, 660, 144)
    canvas.paste(_c30, (726, 2405), _c30)
except Exception:
    pass
layout["Date"] = [726, 2405, 1386, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/31_icon_Relevance.png
try:
    _c31 = get_crop(31, 660, 144)
    canvas.paste(_c31, (54, 2405), _c31)
except Exception:
    pass
layout["Relevance"] = [54, 2405, 714, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/32_icon_Apply_filters.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 2768), _c32)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/33_icon_7.52.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (12, 72), _c33)
except Exception:
    pass
layout["7.52"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/34_icon_7.52.png
try:
    _c34 = get_crop(34, 64, 65)
    canvas.paste(_c34, (112, 1), _c34)
except Exception:
    pass
layout["7.52"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/35_icon_7.52.png
try:
    _c35 = get_crop(35, 61, 64)
    canvas.paste(_c35, (180, 1), _c35)
except Exception:
    pass
layout["7.52"] = [180, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/36_icon_icon_36.png
try:
    _c36 = get_crop(36, 65, 62)
    canvas.paste(_c36, (308, 3), _c36)
except Exception:
    pass
layout["icon_36"] = [308, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/37_icon_Clear_all.png
try:
    _c37 = get_crop(37, 55, 68)
    canvas.paste(_c37, (1319, 0), _c37)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/38_icon_Clear_all.png
try:
    _c38 = get_crop(38, 101, 69)
    canvas.paste(_c38, (1211, 0), _c38)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/39_icon_icon_39.png
try:
    _c39 = get_crop(39, 51, 62)
    canvas.paste(_c39, (248, 2), _c39)
except Exception:
    pass
layout["icon_39"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/40_icon_clickable_35.png
try:
    _c40 = get_crop(40, 144, 144)
    canvas.paste(_c40, (1248, 2110), _c40)
except Exception:
    pass
layout["clickable_35"] = [1248, 2110, 1392, 2254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/41_icon_Clear_all.png
try:
    _c41 = get_crop(41, 178, 144)
    canvas.paste(_c41, (1214, 72), _c41)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/42_icon_Show_all_languages.png
try:
    _c42 = get_crop(42, 511, 144)
    canvas.paste(_c42, (0, 1791), _c42)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1791, 511, 1935]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/43_text_7.52.png
try:
    _c43 = get_crop(43, 91, 45)
    canvas.paste(_c43, (20, 15), _c43)
except Exception:
    pass
layout["7.52"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/44_text_Filters.png
try:
    _c44 = get_crop(44, 180, 66)
    canvas.paste(_c44, (631, 116), _c44)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/45_text_Categories.png
try:
    _c45 = get_crop(45, 187, 135)
    canvas.paste(_c45, (36, 383), _c45)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/46_text_Show_all_categories.png
try:
    _c46 = get_crop(46, 516, 144)
    canvas.paste(_c46, (0, 518), _c46)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/47_text_Event_type.png
try:
    _c47 = get_crop(47, 298, 127)
    canvas.paste(_c47, (36, 829), _c47)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/48_text_Show_less_event_types.png
try:
    _c48 = get_crop(48, 569, 144)
    canvas.paste(_c48, (0, 1345), _c48)
except Exception:
    pass
layout["Show_less_event_types"] = [0, 1345, 569, 1489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/49_text_Languages.png
try:
    _c49 = get_crop(49, 210, 135)
    canvas.paste(_c49, (36, 1656), _c49)
except Exception:
    pass
layout["Languages"] = [36, 1656, 246, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/50_text_Price.png
try:
    _c50 = get_crop(50, 149, 63)
    canvas.paste(_c50, (45, 1995), _c50)
except Exception:
    pass
layout["Price"] = [45, 1995, 194, 2058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/51_text_Only_free_events.png
try:
    _c51 = get_crop(51, 660, 144)
    canvas.paste(_c51, (54, 2405), _c51)
except Exception:
    pass
layout["Only_free_events"] = [54, 2405, 714, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_06_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-8/52_text_Sort_by.png
try:
    _c52 = get_crop(52, 204, 74)
    canvas.paste(_c52, (42, 2313), _c52)
except Exception:
    pass
layout["Sort_by"] = [42, 2313, 246, 2387]
