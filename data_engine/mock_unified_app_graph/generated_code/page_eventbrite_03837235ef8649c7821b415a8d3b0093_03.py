# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_03
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5.png
# step_index: 3/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the filters page using existing canvas & draw objects.

# Colors
BG = "#FFFFFF"
STATUS_BG = "#E9E9E9"
HEADER_BG = "#FFFFFF"
CARD_BG = "#F6FBFF"        # very light blue card backing
CARD_BG_ALT = "#FBFBFC"    # alternative very light card
SHADOW = "#E6E1E8"
DIVIDER = "#E6E0E8"
SEG_RIGHT = "#EDE9EE"
APPLY_BORDER = "#BFB7BE"

w, h = canvas.size

# Fill whole background
draw.rectangle([(0, 0), (w, h)], fill=BG)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BG)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = status_h + 96
draw.rectangle([(0, header_top), (w, header_bottom)], fill=HEADER_BG)
# subtle divider under header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=DIVIDER, width=1)

# Helper to draw card with shadow
def draw_card(x0, y0, x1, y1, fill=CARD_BG, radius=28, shadow_offset=6):
    # shadow
    draw.rounded_rectangle([x0, y0+shadow_offset, x1, y1+shadow_offset], radius=radius, fill=SHADOW)
    # card
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=None)

# Categories group card (background rounded container behind chips)
cat_left = 36
cat_right = w - 36
cat_top = 220
cat_bottom = 700
draw_card(cat_left, cat_top, cat_right, cat_bottom, fill=CARD_BG, radius=34)

# Divider below categories
draw.line([(36 + 8, cat_bottom + 12), (w - 36 - 8, cat_bottom + 12)], fill=DIVIDER, width=1)

# Event type group card
etype_top = 720
etype_bottom = 1120
draw_card(cat_left, etype_top, cat_right, etype_bottom, fill=CARD_BG_ALT, radius=34)

# Divider below event type
draw.line([(36 + 8, etype_bottom + 12), (w - 36 - 8, etype_bottom + 12)], fill=DIVIDER, width=1)

# Languages group card
lang_top = 1180
lang_bottom = 1560
draw_card(cat_left, lang_top, cat_right, lang_bottom, fill=CARD_BG, radius=34)

# Divider below languages
draw.line([(36 + 8, lang_bottom + 12), (w - 36 - 8, lang_bottom + 12)], fill=DIVIDER, width=1)

# Price area (simple subtle card)
price_top = 1600
price_bottom = 1760
draw_card(cat_left, price_top, cat_right, price_bottom, fill=CARD_BG_ALT, radius=20)

# "Only free events" toggle row area (leave toggle itself for overlay; draw subtle background)
toggle_row_top = 1880
toggle_row_bottom = 2080
# Slight background band to group price/toggle/sort controls
draw.rectangle([(36, toggle_row_top), (w-36, toggle_row_bottom)], fill=BG)

# Sort by segmented control background
seg_left = 54
seg_right_x = w - 54
seg_top = 1930
seg_bottom = 2030
seg_radius = 18
# overall container (border + slight shadow)
draw.rounded_rectangle([seg_left, seg_top, seg_right_x, seg_bottom], radius=seg_radius+2, fill=SHADOW)
draw.rounded_rectangle([seg_left, seg_top, seg_right_x, seg_bottom], radius=seg_radius, fill="#FFFFFF", outline=DIVIDER, width=1)

# Two segments: left active (white), right inactive (light grey)
# compute midpoint subtracting small gap for seam
mid_x = (seg_left + seg_right_x) // 2
# left segment (slightly inset to show pressed state)
draw.rounded_rectangle([seg_left+2, seg_top+2, mid_x-1, seg_bottom-2], radius=seg_radius-2, fill="#FFFFFF", outline=None)
# right segment (inactive)
draw.rounded_rectangle([mid_x+1, seg_top+2, seg_right_x-2, seg_bottom-2], radius=seg_radius-2, fill=SEG_RIGHT, outline=None)

# subtle inner divider between segments
draw.line([(mid_x, seg_top+8), (mid_x, seg_bottom-8)], fill=DIVIDER, width=1)

# Bottom "Apply filters" button area (outlined rounded rectangle)
apply_left = 48
apply_right = w - 48
apply_top = 2768
apply_bottom = apply_top + 140
apply_radius = 12
# shadow
draw.rounded_rectangle([apply_left, apply_top+6, apply_right, apply_bottom+6], radius=apply_radius+2, fill=SHADOW)
# button body
draw.rounded_rectangle([apply_left, apply_top, apply_right, apply_bottom], radius=apply_radius, fill=BG, outline=APPLY_BORDER, width=4)

# Top/between separators (subtle)
sep_positions = [header_bottom + 16, cat_top - 16, etype_top - 16, lang_top - 16, price_top - 16, toggle_row_top - 16]
for y in sep_positions:
    draw.line([(36, y), (w-36, y)], fill=DIVIDER, width=1)

# Small bottom safe area line
draw.line([(0, h-1), (w, h-1)], fill=DIVIDER, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/03_icon_Business.png
try:
    _c3 = get_crop(3, 241, 135)
    canvas.paste(_c3, (247, 383), _c3)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/04_icon_French.png
try:
    _c4 = get_crop(4, 205, 144)
    canvas.paste(_c4, (768, 1275), _c4)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/08_icon_Arts.png
try:
    _c8 = get_crop(8, 152, 144)
    canvas.paste(_c8, (1166, 383), _c8)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/09_icon_Seminar.png
try:
    _c9 = get_crop(9, 232, 144)
    canvas.paste(_c9, (358, 829), _c9)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/18_icon_4.41.png
try:
    _c18 = get_crop(18, 61, 65)
    canvas.paste(_c18, (179, 1), _c18)
except Exception:
    pass
layout["4.41"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/19_icon_4.41.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["4.41"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/20_icon_4.41.png
try:
    _c20 = get_crop(20, 66, 66)
    canvas.paste(_c20, (110, 1), _c20)
except Exception:
    pass
layout["4.41"] = [110, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 65, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/25_icon_Toggle_to_filter_only_free_events.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/27_text_4.41.png
try:
    _c27 = get_crop(27, 87, 43)
    canvas.paste(_c27, (22, 15), _c27)
except Exception:
    pass
layout["4.41"] = [22, 15, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_03_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
