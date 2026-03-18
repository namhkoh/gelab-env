# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_05
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7.png
# step_index: 5/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Drawstructural UI elements for the Filters screen using provided canvas and draw objects.
# Uses only canvas and draw (no imports). Avoids drawing any detected icons/text elements.

# Colors
STATUS_BAR = "#dcdcdc"      # light gray status bar
HEADER_BG = "#ffffff"       # header background (white)
DIVIDER = "#e6e6ea"         # subtle divider lines
CARD_BG = "#fbfcff"         # very light card background
CARD_BORDER = "#e9e9ee"     # card border
SHADOW = "#f0f0f5"          # subtle shadow tone
PAGE_BG = "#ffffff"         # page background (canvas already white)

W = canvas.width
H = canvas.height

# Helper to draw rounded rectangle with optional border
def rounded_card(rect, radius=20, fill=CARD_BG, outline=CARD_BORDER, outline_width=1):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=outline_width)

# Top status bar
status_h = 88
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BAR)

# Header area (below status bar)
header_top = status_h
header_bottom = 170
draw.rectangle([(0, header_top), (W, header_bottom)], fill=HEADER_BG)

# Header subtle bottom divider / shadow
draw.line([(36, header_bottom), (W-36, header_bottom)], fill=DIVIDER, width=1)
# Slight subtle shadow line above divider for depth
draw.line([(36, header_bottom-2), (W-36, header_bottom-2)], fill=SHADOW, width=1)

# Main content cards / section backgrounds
left_margin = 36
right_margin = W - 36

# Categories card (background behind the category chips)
cat_top = header_bottom + 10    # ~180
cat_bottom = 580
rounded_card((left_margin, cat_top, right_margin, cat_bottom), radius=20)

# Event type card
etype_top = cat_bottom + 20     # ~600
etype_bottom = 980
rounded_card((left_margin, etype_top, right_margin, etype_bottom), radius=20)

# Languages card
lang_top = etype_bottom + 20    # ~1000
lang_bottom = 1400
rounded_card((left_margin, lang_top, right_margin, lang_bottom), radius=20)

# Price area card (simple card behind price and toggle)
price_top = lang_bottom + 20    # ~1420
price_bottom = 1760
rounded_card((left_margin, price_top, right_margin, price_bottom), radius=16)

# Sort by container (background for segmented control)
sort_top = 1860
sort_bottom = 2180
# Draw a slightly darker background to separate the sort control area
draw.rounded_rectangle((left_margin, sort_top, right_margin, sort_bottom),
                       radius=18, fill="#f7f7fa", outline="#e6e6ea", width=1)

# Add subtle inner shadow under some cards to give depth
def inner_shadow(rect, offset=6, opacity_line=1):
    x1, y1, x2, y2 = rect
    # bottom shadow line
    draw.line([(x1+8, y2-offset), (x2-8, y2-offset)], fill=SHADOW, width=1)

inner_shadow((left_margin, cat_top, right_margin, cat_bottom))
inner_shadow((left_margin, etype_top, right_margin, etype_bottom))
inner_shadow((left_margin, lang_top, right_margin, lang_bottom))
inner_shadow((left_margin, price_top, right_margin, price_bottom))
inner_shadow((left_margin, sort_top, right_margin, sort_bottom))

# Horizontal separators between logical sections (subtle)
sep_positions = [
    cat_bottom + 10,
    etype_bottom + 10,
    lang_bottom + 10,
    price_bottom + 10,
    sort_bottom + 10
]
for y in sep_positions:
    draw.line([(left_margin, y), (right_margin, y)], fill=DIVIDER, width=1)

# Bottom area: leave space for "Apply filters" element (detected and will be pasted).
# Draw a faint top border above that area to separate content from the button (but do not draw inside the button rect).
apply_top = 2768
# Draw a subtle divider a little above the apply button area
draw.line([(left_margin, apply_top-28), (right_margin, apply_top-28)], fill=DIVIDER, width=1)

# Final subtle vignette at very bottom (very light)
draw.rectangle([(0, apply_top+160), (W, H)], fill="#ffffff")

# Done. All drawn elements are backgrounds/structure only; no text or icons were rendered.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/18_icon_7.52.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.52"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/19_icon_7.52.png
try:
    _c19 = get_crop(19, 61, 63)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["7.52"] = [179, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 66, 62)
    canvas.paste(_c20, (307, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/21_icon_7.52.png
try:
    _c21 = get_crop(21, 64, 65)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["7.52"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1375, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/25_icon_Click_to_toggle_filtering_for_only_free_.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Click_to_toggle_filtering"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/27_text_7.52.png
try:
    _c27 = get_crop(27, 91, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["7.52"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_05_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-7/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
