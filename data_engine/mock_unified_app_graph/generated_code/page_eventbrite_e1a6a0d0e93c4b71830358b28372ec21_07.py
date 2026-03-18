# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_07
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9.png
# step_index: 7/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the filter UI (PIL drawing)
# Uses provided variables: canvas (Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 251, 252)        # very light page background (almost white)
status_bar_color = (190, 190, 190)  # status bar grey
divider_color = (230, 230, 235)   # subtle divider
card_fill = (255, 255, 255)       # white card fill (helps subtle cards stand out on off-white)
card_shadow = (235, 235, 240)     # very light shadow
muted_grey = (245, 246, 248)      # very faint panel fill
bottom_bar_bg = (248, 248, 250)   # bottom area behind apply filters
accent_shadow = (220, 220, 225)   # slightly darker for inner shadows

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header area (toolbar) - leave icons/text space clear, but draw subtle divider below
header_top = status_h
header_h = 72
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill=card_fill)
draw.line([(24, header_top + header_h - 1), (W - 24, header_top + header_h - 1)], fill=divider_color, width=2)

# Helper for subtle shadowed rounded card
def draw_shadowed_card(box, radius=18, fill=card_fill, shadow_color=card_shadow, shadow_offset=(0, 6)):
    x1, y1, x2, y2 = box
    ox, oy = shadow_offset
    # shadow
    shadow_box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
    draw.rounded_rectangle(shadow_box, radius=radius, fill=shadow_color)
    # main card
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=None)

# Section card backgrounds (groups) - behind chips / lists (do not draw chips or texts)
# Wide margins similar to UI
left = 24
right = W - 24

# Categories group background (subtle, behind category chips)
cat_top = 320
cat_bottom = 520
draw_shadowed_card((left, cat_top, right, cat_bottom), radius=28, fill=muted_grey, shadow_color=card_shadow, shadow_offset=(0,4))

# Event type group background
etype_top = 740
etype_bottom = 980
draw_shadowed_card((left, etype_top, right, etype_bottom), radius=28, fill=muted_grey, shadow_color=card_shadow, shadow_offset=(0,4))

# Languages group background
lang_top = 1180
lang_bottom = 1460
draw_shadowed_card((left, lang_top, right, lang_bottom), radius=28, fill=muted_grey, shadow_color=card_shadow, shadow_offset=(0,4))

# Price / Toggle area (compact background block)
price_top = 1548
price_bottom = 1708
# a faint panel rather than a full card to keep toggle area airy
draw.rounded_rectangle([left, price_top, right, price_bottom], radius=16, fill=card_fill, outline=divider_color)

# Sort by area - create a faint container but do NOT draw the segmented control itself
sort_top = 1888
sort_bottom = 2068
# shadow + container
draw_shadowed_card((left, sort_top, right, sort_bottom), radius=14, fill=muted_grey, shadow_color=accent_shadow, shadow_offset=(0,3))

# Subtle separators between logical sections (thin lines)
sep_positions = [
    header_top + header_h + 12,  # under header
    600,   # between categories and event type flow
    1080,  # between event type and languages
    1550,  # above Price area
    1860,  # above Sort by
]
for y in sep_positions:
    draw.line([(24, y), (W - 24, y)], fill=divider_color, width=1)

# Bottom area behind the "Apply filters" control (keep it distinct but don't redraw the button)
bottom_bg_top = 2680
bottom_bg_rect = (12, bottom_bg_top, W - 12, H)
draw.rounded_rectangle(bottom_bg_rect, radius=16, fill=bottom_bar_bg, outline=accent_shadow)

# Add an outer subtle border near the bottom to emphasize safe area (not the button)
draw.rectangle([(12, bottom_bg_top), (W - 12, bottom_bg_top + 2)], fill=divider_color)

# Light left/right edge shadows to give depth to the page (very subtle)
edge_shadow_color = (245, 245, 247)
draw.rectangle([(0, header_top + 1), (12, H - 1)], fill=edge_shadow_color)
draw.rectangle([(W - 12, header_top + 1), (W, H - 1)], fill=edge_shadow_color)

# Top header left/back area - keep empty (icons/text will be pasted). Add a tiny chevron background area to indicate tappable zone (very subtle)
chev_zone = (24, header_top + 10, 84, header_top + 62)
draw.rounded_rectangle(chev_zone, radius=12, fill=muted_grey)

# Right "clear all" area tap zone (subtle highlight, not an icon)
clear_zone = (W - 84, header_top + 10, W - 24, header_top + 62)
draw.rounded_rectangle(clear_zone, radius=12, fill=muted_grey)

# Final gentle vignette at very bottom to ground the page (very subtle)
vignette_top = H - 140
draw.rectangle([(0, vignette_top), (W, H)], fill=(252, 252, 253))

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/18_icon_5.18.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["5.18"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/19_icon_5.18.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (180, 1), _c19)
except Exception:
    pass
layout["5.18"] = [180, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 64, 62)
    canvas.paste(_c20, (308, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/21_icon_5.18.png
try:
    _c21 = get_crop(21, 64, 66)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["5.18"] = [112, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/27_text_5.18.png
try:
    _c27 = get_crop(27, 91, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["5.18"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_07_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-9/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
