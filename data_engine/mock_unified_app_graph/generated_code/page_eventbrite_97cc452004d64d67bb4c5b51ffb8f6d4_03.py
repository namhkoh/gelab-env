# page_id: page_eventbrite_97cc452004d64d67bb4c5b51ffb8f6d4_03
# screenshot: 2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5.png
# step_index: 3/7
# task: Open Eventbrite. Search Business event. Select the first one that is not promoted. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (slightly off-white to match the app background)
draw.rectangle((0, 0, 1440, 2960), fill="#FBFCFF")

# Status bar area (top ~88px) - light grey background
status_h = 88
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Thin darker strip at the very top to emulate device bezel shadow
draw.rectangle((0, 0, 1440, 6), fill="#B7B7B7")

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 180
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")

# Subtle divider / shadow under the header
draw.line((36, header_bottom - 6, 1404, header_bottom - 6), fill="#E9EAEE", width=2)
draw.line((36, header_bottom - 3, 1404, header_bottom - 3), fill="#F6F6F8", width=1)

# Main content container background (a very faint card-like area)
content_top = header_bottom + 8
content_bottom = 2720  # stop above the bottom action bar area (which will be pasted)
# Draw a light shadow first (slightly offset)
shadow_rect = (28, content_top + 8, 1412, content_bottom + 8)
draw.rounded_rectangle(shadow_rect, radius=12, fill="#F1F2F6")
# Then draw the actual container (slightly inset)
content_rect = (36, content_top, 1404, content_bottom)
draw.rounded_rectangle(content_rect, radius=12, fill="#FFFFFF")

# Section separator lines (subtle) placed between groups of filters
# Keep separators out of areas where detected elements will be pasted.
sep_color = "#F0F1F5"
# Separator after categories area (below "Show all categories")
draw.line((72, 690, 1368, 690), fill=sep_color, width=1)
# Separator after event type area
draw.line((72, 1140, 1368, 1140), fill=sep_color, width=1)
# Separator after languages section
draw.line((72, 1580, 1368, 1580), fill=sep_color, width=1)
# Subtle divider above sort/toggle area
draw.line((72, 1840, 1368, 1840), fill=sep_color, width=1)

# Add a soft inset shadow along the left and right edges of the content card for depth
for i, a in enumerate([0, 1, 2]):
    alpha = 1 + i
    # left edge
    draw.line((36 + i, content_top + 6, 36 + i, content_bottom - 6), fill="#F3F4F6")
    # right edge
    draw.line((1404 - i, content_top + 6, 1404 - i, content_bottom - 6), fill="#F3F4F6")

# Top area accent: a subtle pale divider band under the header inside the content area
draw.rectangle((36, content_top, 1404, content_top + 6), fill="#FBFBFD")

# Bottom safe area above the "Apply filters" action (leave exact action area untouched)
# Draw a faint top border to separate content from the action area that will be pasted
apply_top = 2768
draw.line((36, apply_top - 12, 1404, apply_top - 12), fill="#E8E9ED", width=2)
draw.line((36, apply_top - 9, 1404, apply_top - 9), fill="#FAFAFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/18_icon_9.39.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.39"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/19_icon_9.39.png
try:
    _c19 = get_crop(19, 64, 64)
    canvas.paste(_c19, (176, 1), _c19)
except Exception:
    pass
layout["9.39"] = [176, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 55, 69)
    canvas.paste(_c20, (1319, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 99, 69)
    canvas.paste(_c21, (1211, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/22_icon_9.39.png
try:
    _c22 = get_crop(22, 59, 65)
    canvas.paste(_c22, (112, 1), _c22)
except Exception:
    pass
layout["9.39"] = [112, 1, 171, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/27_text_9.39.png
try:
    _c27 = get_crop(27, 96, 49)
    canvas.paste(_c27, (16, 12), _c27)
except Exception:
    pass
layout["9.39"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_03_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
