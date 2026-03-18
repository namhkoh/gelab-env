# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_17
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19.png
# step_index: 17/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))

# Header area (toolbar)
header_top = status_h
header_h = 96
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill=(255, 255, 255))

# Thin divider below header
divider_y = header_top + header_h
draw.line([(24, divider_y), (1440-24, divider_y)], fill=(236, 236, 240), width=2)

# Section background: Categories block
cat_top = divider_y + 44
cat_bottom = cat_top + 300
draw.rounded_rectangle([(36, cat_top), (1440-36, cat_bottom)], radius=18, fill=(250, 251, 255), outline=(240,240,245), width=1)

# Separator under categories
sep1 = cat_bottom + 18
draw.line([(36, sep1), (1440-36, sep1)], fill=(245,245,248), width=1)

# Section background: Event type block
etype_top = sep1 + 44
etype_bottom = etype_top + 260
draw.rounded_rectangle([(36, etype_top), (1440-36, etype_bottom)], radius=18, fill=(250, 251, 255), outline=(240,240,245), width=1)

# Separator under event type
sep2 = etype_bottom + 18
draw.line([(36, sep2), (1440-36, sep2)], fill=(245,245,248), width=1)

# Section background: Languages block
lang_top = sep2 + 44
lang_bottom = lang_top + 260
draw.rounded_rectangle([(36, lang_top), (1440-36, lang_bottom)], radius=18, fill=(250, 251, 255), outline=(240,240,245), width=1)

# Separator under languages
sep3 = lang_bottom + 18
draw.line([(36, sep3), (1440-36, sep3)], fill=(245,245,248), width=1)

# Price / Toggle area background (subtle)
price_top = sep3 + 44
price_bottom = price_top + 220
draw.rounded_rectangle([(36, price_top), (1440-36, price_bottom)], radius=14, fill=(255,255,255), outline=(245,245,248), width=1)

# Separator above sort-by area
sep4 = price_bottom + 20
draw.line([(36, sep4), (1440-36, sep4)], fill=(245,245,248), width=1)

# Sort-by container background (rounded pill container behind tabs)
sort_top = sep4 + 36
sort_h = 84
sort_left = 48
sort_right = 1440 - 48
draw.rounded_rectangle([(sort_left, sort_top), (sort_right, sort_top + sort_h)], radius=18, fill=(246,244,249), outline=(232,229,235), width=1)

# Large empty content area (keep mostly white)
content_top = sort_top + sort_h + 40
draw.rectangle([(0, content_top), (1440, 2560)], fill=(255,255,255))

# Bottom apply-filters area background shadow (above footer)
footer_top = 2660
draw.rectangle([(0, footer_top), (1440, 2960)], fill=(254,254,255))
# subtle top border for footer area
draw.line([(36, footer_top), (1440-36, footer_top)], fill=(235,235,240), width=2)

# Add subtle vertical guides at left/right margins
draw.line([(36, header_top), (36, 2560)], fill=(250,250,252), width=1)
draw.line([(1440-36, header_top), (1440-36, 2560)], fill=(250,250,252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 135)
    canvas.paste(_c0, (36, 383), _c0)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/02_icon_French.png
try:
    _c2 = get_crop(2, 205, 144)
    canvas.paste(_c2, (768, 1275), _c2)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/18_icon_Clear_all.png
try:
    _c18 = get_crop(18, 52, 68)
    canvas.paste(_c18, (1153, 1), _c18)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1205, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/19_icon_7.48.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["7.48"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/20_icon_7.48.png
try:
    _c20 = get_crop(20, 61, 64)
    canvas.paste(_c20, (179, 2), _c20)
except Exception:
    pass
layout["7.48"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 98, 68)
    canvas.paste(_c21, (1211, 1), _c21)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1309, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/22_icon_7.48.png
try:
    _c22 = get_crop(22, 65, 65)
    canvas.paste(_c22, (111, 1), _c22)
except Exception:
    pass
layout["7.48"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 64, 62)
    canvas.paste(_c23, (308, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 53, 67)
    canvas.paste(_c24, (1319, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 61)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/27_icon_Toggle_to_filter_only_free_events.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/28_text_7.48.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 13), _c28)
except Exception:
    pass
layout["7.48"] = [20, 13, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_17_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-19/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
