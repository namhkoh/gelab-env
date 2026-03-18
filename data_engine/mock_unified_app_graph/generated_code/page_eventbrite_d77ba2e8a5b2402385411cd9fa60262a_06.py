# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_06
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8.png
# step_index: 6/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (slight off-white to match screenshot)
bg_color = (255, 255, 255)
draw.rectangle([0, 0, canvas.width, canvas.height], fill=bg_color)

# Colors
status_bar_color = (196, 196, 196)    # light gray for status bar
header_bg = (255, 255, 255)           # white header
divider_color = (230, 230, 235)       # subtle divider
card_fill = (245, 250, 255)           # very light bluish for grouped section backgrounds
card_border = (220, 225, 235)         # faint border for cards
shadow_color = (220, 220, 220)

# Status bar (top)
status_h = 96
draw.rectangle([0, 0, canvas.width, status_h], fill=status_bar_color)

# Header / toolbar area below status bar
header_top = status_h
header_bot = 200
draw.rectangle([0, header_top, canvas.width, header_bot], fill=header_bg)
# subtle shadow/divider under header
draw.line([24, header_bot, canvas.width - 24, header_bot], fill=divider_color, width=2)

# Draw faint shadow line at bottom of status bar to separate icons area
draw.line([0, status_h, canvas.width, status_h], fill=shadow_color, width=1)

# Section grouped background cards (rounded rectangles behind groups)
# Categories card (behind chips area)
cat_card_left = 24
cat_card_top = 220
cat_card_right = canvas.width - 24
cat_card_bottom = 560
draw.rounded_rectangle([cat_card_left, cat_card_top, cat_card_right, cat_card_bottom],
                       radius=28, fill=card_fill, outline=card_border, width=1)

# Event type card
etype_card_top = 700
etype_card_bottom = 1040
draw.rounded_rectangle([cat_card_left, etype_card_top, cat_card_right, etype_card_bottom],
                       radius=28, fill=card_fill, outline=card_border, width=1)

# Languages card
lang_card_top = 1140
lang_card_bottom = 1480
draw.rounded_rectangle([cat_card_left, lang_card_top, cat_card_right, lang_card_bottom],
                       radius=28, fill=card_fill, outline=card_border, width=1)

# Price area subtle background (small rounded rect grouping the Price and toggle area)
price_card_top = 1560
price_card_bottom = 2100
# Make it very subtle and narrow width to avoid overlapping bottom controls
draw.rounded_rectangle([36, price_card_top, canvas.width - 36, price_card_bottom],
                       radius=18, fill=(255, 255, 255), outline=(245, 245, 248), width=1)

# Separator lines between major logical sections (thin)
separators = [
    600,   # below categories
    1080,  # below event type
    1520,  # below languages
    1700,  # end of price area (above sort by)
]
for y in separators:
    draw.line([36, y, canvas.width - 36, y], fill=divider_color, width=1)

# Light rounded background behind the "Sort by" control area (so pasted control sits on it)
sort_area_top = 1880
sort_area_bottom = 2060
draw.rounded_rectangle([36, sort_area_top, canvas.width - 36, sort_area_bottom],
                       radius=12, fill=(250, 250, 252), outline=(235, 235, 240), width=1)

# Subtle bottom safe-area background strip (above Apply filters bar)
bottom_strip_top = canvas.height - 250
draw.rectangle([0, bottom_strip_top, canvas.width, canvas.height], fill=(255, 255, 255))

# Add a faint outline around the entire content area to give subtle structure
draw.rectangle([12, header_bot + 6, canvas.width - 12, canvas.height - 220], outline=(245,245,248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/12_icon_English.png
try:
    _c12 = get_crop(12, 210, 135)
    canvas.paste(_c12, (36, 1275), _c12)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/13_icon_German.png
try:
    _c13 = get_crop(13, 225, 135)
    canvas.paste(_c13, (270, 1275), _c13)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/18_icon_6.49.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["6.49"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/19_icon_6.49.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["6.49"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/20_icon_6.49.png
try:
    _c20 = get_crop(20, 65, 66)
    canvas.paste(_c20, (111, 1), _c20)
except Exception:
    pass
layout["6.49"] = [111, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 64, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 99, 65)
    canvas.paste(_c22, (1211, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 56, 67)
    canvas.paste(_c23, (1317, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 61)
    canvas.paste(_c24, (248, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/27_text_6.49.png
try:
    _c27 = get_crop(27, 89, 43)
    canvas.paste(_c27, (22, 15), _c27)
except Exception:
    pass
layout["6.49"] = [22, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_06_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-8/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
