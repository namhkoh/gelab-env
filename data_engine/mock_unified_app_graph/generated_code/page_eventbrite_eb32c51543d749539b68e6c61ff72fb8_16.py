# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_16
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18.png
# step_index: 16/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas is provided (1440x2960) as `canvas`, `draw` is an ImageDraw object.
# Fonts available: font_sm, font_md, font_lg, font_xl

# ---- Colors ----
bg_color = "#fbfcfe"         # very light off-white background
status_color = "#cfcfcf"     # status bar grey
header_color = "#ffffff"     # toolbar/header white
divider_color = "#e7e7ea"    # subtle divider lines
card_fill = "#f6fbff"        # very light bluish card background
card_outline = "#e6eaef"     # card outline / subtle border
shadow_color = "#ececf0"     # faint shadow under cards

# ---- Fill overall background ----
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# ---- Status bar (top) ----
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# subtle darker line at bottom of status bar
draw.line([(0, status_h), (1440, status_h)], fill=divider_color, width=1)

# ---- Header / Toolbar ----
header_top = status_h
header_bottom = status_h + 124  # header area height
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_color)

# header bottom divider
draw.line([(24, header_bottom), (1416, header_bottom)], fill=divider_color, width=1)

# slight shadow under header
draw.rectangle([(0, header_bottom), (1440, header_bottom+4)], fill=shadow_color)

# ---- Section card backgrounds (rounded rectangles behind groups) ----
pad_x = 36
card_radius = 18

# Categories group
cat_top = 200
cat_bottom = 540
draw.rounded_rectangle(
    [(pad_x, cat_top), (1440 - pad_x, cat_bottom)],
    radius=card_radius, fill=card_fill, outline=card_outline, width=1
)
# faint inner shadow line
draw.line([(pad_x + 2, cat_bottom), (1440 - pad_x - 2, cat_bottom)], fill=shadow_color, width=1)

# Event type group
etype_top = 660
etype_bottom = 1000
draw.rounded_rectangle(
    [(pad_x, etype_top), (1440 - pad_x, etype_bottom)],
    radius=card_radius, fill=card_fill, outline=card_outline, width=1
)
draw.line([(pad_x + 2, etype_bottom), (1440 - pad_x - 2, etype_bottom)], fill=shadow_color, width=1)

# Languages group
lang_top = 1090
lang_bottom = 1430
draw.rounded_rectangle(
    [(pad_x, lang_top), (1440 - pad_x, lang_bottom)],
    radius=card_radius, fill=card_fill, outline=card_outline, width=1
)
draw.line([(pad_x + 2, lang_bottom), (1440 - pad_x - 2, lang_bottom)], fill=shadow_color, width=1)

# Price group (heading area with subtle background)
price_top = 1520
price_bottom = 1710
draw.rounded_rectangle(
    [(pad_x, price_top), (1440 - pad_x, price_bottom)],
    radius=card_radius, fill="#ffffff", outline=card_outline, width=1
)
draw.line([(pad_x + 2, price_bottom), (1440 - pad_x - 2, price_bottom)], fill=shadow_color, width=1)

# Sort / Toggle group (holds sort control, toggle, etc.)
sort_top = 1840
sort_bottom = 2100
draw.rounded_rectangle(
    [(pad_x, sort_top), (1440 - pad_x, sort_bottom)],
    radius=card_radius, fill="#ffffff", outline=card_outline, width=1
)
draw.line([(pad_x + 2, sort_bottom), (1440 - pad_x - 2, sort_bottom)], fill=shadow_color, width=1)

# ---- Additional separators between major sections ----
sep_color = divider_color
separator_x0 = pad_x
separator_x1 = 1440 - pad_x

# Between categories and event types
draw.line([(separator_x0, cat_bottom + 40), (separator_x1, cat_bottom + 40)], fill=sep_color, width=1)

# Between event types and languages
draw.line([(separator_x0, etype_bottom + 40), (separator_x1, etype_bottom + 40)], fill=sep_color, width=1)

# Between languages and price
draw.line([(separator_x0, lang_bottom + 40), (separator_x1, lang_bottom + 40)], fill=sep_color, width=1)

# Between price and sort
draw.line([(separator_x0, price_bottom + 40), (separator_x1, price_bottom + 40)], fill=sep_color, width=1)

# ---- Subtle left/right page margins shading ----
margin_shade = "#fafbfc"
draw.rectangle([(0, header_bottom+1), (pad_x, 2960)], fill=margin_shade)
draw.rectangle([(1440 - pad_x, header_bottom+1), (1440, 2960)], fill=margin_shade)

# ---- Bottom area safe shadow (above the bottom action bar which will be pasted externally) ----
# Draw a soft shadow band to indicate separation from the main content to stuck action area
bottom_shadow_top = 2700
bottom_shadow_bottom = 2768
draw.rectangle([(24, bottom_shadow_top), (1440 - 24, bottom_shadow_bottom)], fill=shadow_color)

# Thin top border for the eventual bottom action button area (do not draw the button itself)
draw.line([(24, bottom_shadow_top), (1440 - 24, bottom_shadow_top)], fill=divider_color, width=2)

# ---- Small decorative accents (non-interactive) ----
# A faint pill-style background behind where the sort control sits (as a grouping hint)
pill_left = 48
pill_top = sort_top + 34
pill_right = 1392
pill_bottom = pill_top + 84
draw.rounded_rectangle(
    [(pill_left, pill_top), (pill_right, pill_bottom)],
    radius=14, fill="#f7f7f9", outline="#efeff2", width=1
)

# End of drawing. The detected UI elements (icons, buttons, and text) will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/19_icon_7.48.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["7.48"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 100, 70)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/21_icon_7.48.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (12, 72), _c21)
except Exception:
    pass
layout["7.48"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/22_icon_7.48.png
try:
    _c22 = get_crop(22, 65, 65)
    canvas.paste(_c22, (111, 1), _c22)
except Exception:
    pass
layout["7.48"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 66, 62)
    canvas.paste(_c23, (307, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 52, 69)
    canvas.paste(_c24, (1320, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 62)
    canvas.paste(_c25, (248, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/27_icon_Toggle_to_filter_only_free_events..png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/28_text_7.48.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 13), _c28)
except Exception:
    pass
layout["7.48"] = [20, 13, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_16_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-18/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
