# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_06
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8.png
# step_index: 6/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Filters page

# Canvas is provided (1440x2960 RGB) and draw is an ImageDraw.Draw instance.
# Fonts font_sm, font_md, font_lg, font_xl are available but we will not draw text.

w, h = canvas.size

# Colors (based on screenshot)
bg_color = (250, 251, 252)         # very light off-white background
status_bar_color = (196, 196, 196) # top status bar gray
header_bg = (255, 255, 255)        # header/toolbar white
divider = (230, 234, 240)          # subtle divider
card_bg = (247, 250, 252)          # slight card background for groups
card_shadow = (220, 224, 232)      # shadow under cards
seg_bg = (245, 244, 249)           # segmented control background
seg_border = (200, 198, 207)       # segmented control border
muted_separator = (235, 238, 242)  # thin separators

# Fill overall background
draw.rectangle([0, 0, w, h], fill=bg_color)

# Status bar area (top)
status_h = 84
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# Header area (toolbar) beneath status bar
header_h = 116  # from top to header title baseline roughly
draw.rectangle([0, status_h, w, header_h], fill=header_bg)

# Header bottom divider
draw.line([(24, header_h), (w-24, header_h)], fill=divider, width=1)

# Soft shadow under header (very light)
draw.line([(24, header_h+1), (w-24, header_h+1)], fill=(245,245,246), width=1)

# Helper to draw rounded rectangles with optional shadow
def rounded_card(x0, y0, x1, y1, radius=20, fill=card_bg, shadow=True):
    if shadow:
        # shadow offset
        so = 6
        draw.rounded_rectangle([x0, y0+so, x1, y1+so], radius=radius, fill=card_shadow)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)

# Section group cards (behind chips and grouped controls)
# Categories group
rounded_card(24, 320, w-24, 540, radius=24, fill=card_bg, shadow=True)

# Event type group
rounded_card(24, 760, w-24, 960, radius=24, fill=card_bg, shadow=True)

# Languages group
rounded_card(24, 1196, w-24, 1456, radius=24, fill=card_bg, shadow=True)

# Price area card (subtle)
rounded_card(24, 1556, w-24, 1696, radius=18, fill=card_bg, shadow=False)

# Sort-by segmented control background container (big rounded rect with subtle border)
seg_x0 = 36
seg_x1 = w - 36
seg_y0 = 2006
seg_y1 = 2170
# outer shadow
draw.rounded_rectangle([seg_x0+4, seg_y0+8, seg_x1+4, seg_y1+8], radius=18, fill=(235,238,242))
# main seg background
draw.rounded_rectangle([seg_x0, seg_y0, seg_x1, seg_y1], radius=18, fill=seg_bg, outline=seg_border, width=2)

# Inner subtle divide for the two segments (visual only — actual buttons will be pasted)
mid = (seg_x0 + seg_x1) // 2
draw.line([(mid, seg_y0+6), (mid, seg_y1-6)], fill=(240,238,244), width=2)

# Separator lines between major sections (subtle)
sep_positions = [
    300,  # above Categories
    700,  # above Event type
    1130, # above Languages
    1520, # above Price
    1960, # above Sort by
]
for y in sep_positions:
    draw.line([(24, y), (w-24, y)], fill=muted_separator, width=1)

# Additional structural accent: faint vertical margin lines near content edges
draw.line([(24, header_h+12), (24, h-180)], fill=(245,247,249), width=1)
draw.line([(w-24, header_h+12), (w-24, h-180)], fill=(245,247,249), width=1)

# Big bottom apply-bar container outline (only structural border, not the button itself)
# Keep it very subtle since the actual "Apply filters" button will be pasted exactly on top.
bottom_bar_y = h - 220
draw.rectangle([24, bottom_bar_y, w-24, h-60], outline=(210,210,216), width=3, fill=(255,255,255,0))

# Thin top shadow for the bottom bar area
draw.line([(24, bottom_bar_y), (w-24, bottom_bar_y)], fill=(240,240,244), width=1)

# Final subtle vignette lines to match screenshot structure (do not resemble icons/text)
draw.line([(24, 220), (w-24, 220)], fill=(248,249,250), width=1)
draw.line([(24, 420), (w-24, 420)], fill=(248,249,250), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/12_icon_English.png
try:
    _c12 = get_crop(12, 210, 135)
    canvas.paste(_c12, (36, 1275), _c12)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/13_icon_German.png
try:
    _c13 = get_crop(13, 225, 135)
    canvas.paste(_c13, (270, 1275), _c13)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/18_icon_7.55.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.55"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/19_icon_7.55.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["7.55"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/20_icon_7.55.png
try:
    _c20 = get_crop(20, 65, 65)
    canvas.paste(_c20, (111, 1), _c20)
except Exception:
    pass
layout["7.55"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 64, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 99, 65)
    canvas.paste(_c22, (1211, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 56, 67)
    canvas.paste(_c23, (1318, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1318, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 61)
    canvas.paste(_c24, (248, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/25_icon_Toggle_to_show_only_free_events..png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_show_only_free_"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/27_text_Filters.png
try:
    _c27 = get_crop(27, 180, 66)
    canvas.paste(_c27, (631, 116), _c27)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/28_text_Categories.png
try:
    _c28 = get_crop(28, 187, 135)
    canvas.paste(_c28, (36, 383), _c28)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/29_text_Show_all_categories.png
try:
    _c29 = get_crop(29, 516, 144)
    canvas.paste(_c29, (0, 518), _c29)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/30_text_Event_type.png
try:
    _c30 = get_crop(30, 298, 135)
    canvas.paste(_c30, (36, 829), _c30)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/31_text_Show_all_event_types.png
try:
    _c31 = get_crop(31, 535, 144)
    canvas.paste(_c31, (0, 964), _c31)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/32_text_Languages.png
try:
    _c32 = get_crop(32, 210, 135)
    canvas.paste(_c32, (36, 1275), _c32)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/33_text_Show_all_languages.png
try:
    _c33 = get_crop(33, 511, 144)
    canvas.paste(_c33, (0, 1410), _c33)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/34_text_Price.png
try:
    _c34 = get_crop(34, 149, 63)
    canvas.paste(_c34, (45, 1613), _c34)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/35_text_Only_free_events.png
try:
    _c35 = get_crop(35, 660, 144)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_06_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-8/36_text_Sort_by.png
try:
    _c36 = get_crop(36, 206, 75)
    canvas.paste(_c36, (42, 1931), _c36)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
