# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_10
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12.png
# step_index: 10/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Filters page
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

width, height = canvas.size

# Colors
status_bar_color = (156, 160, 166)      # muted gray for status bar
divider_color = (230, 230, 235)         # light divider
muted_line = (240, 239, 242)            # very light separator
segment_bg = (246, 245, 249)            # faint segment background
shadow_color = (225, 223, 229)          # subtle shadow
page_bg = (255, 255, 255)               # page white (canvas already white)

# Fill full background (safe—keeps consistent base)
draw.rectangle([(0, 0), (width, height)], fill=page_bg)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (width, status_h)], fill=status_bar_color)

# Header / toolbar area (below status bar)
header_y0 = status_h
header_y1 = 168
# keep header white (same as page) but draw a subtle bottom divider
draw.rectangle([(0, header_y0), (width, header_y1)], fill=page_bg)
draw.line([(0, header_y1), (width, header_y1)], fill=divider_color, width=1)

# Subtle horizontal separators between major filter sections
# Categories section roughly ends around y ~700
sep_positions = [
    700,   # after Categories
    1120,  # after Event type
    1580,  # after Languages / Price area
    1900   # above Sort by area
]
for y in sep_positions:
    draw.line([(36, y), (width-36, y)], fill=muted_line, width=1)

# Soft background for the Sort-by control area (light rounded rect)
sort_area_top = 1930
sort_area_bottom = 2090
sort_left = 40
sort_right = width - 40
# rounded rectangle for visual grouping (very pale)
try:
    draw.rounded_rectangle(
        [(sort_left, sort_area_top), (sort_right, sort_area_bottom)],
        radius=18, fill=segment_bg, outline=shadow_color, width=1
    )
except Exception:
    # Fallback: draw rectangle + thin outline if rounded not available
    draw.rectangle([(sort_left, sort_area_top), (sort_right, sort_area_bottom)], fill=segment_bg)
    draw.rectangle([(sort_left, sort_area_top), (sort_right, sort_area_bottom)], outline=shadow_color, width=1)

# Very subtle drop shadow line below the sort area
draw.line([(sort_left+6, sort_area_bottom+6), (sort_right-6, sort_area_bottom+6)], fill=shadow_color, width=1)

# Light divider above the "Apply filters" button area (to visually separate content from the fixed button)
# We keep a margin so we don't draw over the actual button crop which will be pasted later.
apply_top = 2768
divider_y = apply_top - 28
draw.line([(36, divider_y), (width-36, divider_y)], fill=muted_line, width=1)

# Add a faint rounded backdrop near the bottom edge to suggest an elevated area above the button
backdrop_top = divider_y - 28
backdrop_bottom = divider_y + 8
try:
    draw.rounded_rectangle(
        [(36, backdrop_top), (width-36, backdrop_bottom)],
        radius=12, fill=(252,252,253), outline=muted_line, width=1
    )
except Exception:
    draw.rectangle([(36, backdrop_top), (width-36, backdrop_bottom)], fill=(252,252,253), outline=muted_line)

# Final small decorative lines for section headings (left-aligned), but do not draw text/icons
heading_x = 36
heading_width = 420
heading_lines = [
    (130, 130 + 0),   # small accent area near top header (no text)
    (518, 518 + 0),   # near "Show all categories" area
    (964, 964 + 0),   # near "Show all event types" area
    (1410, 1410 + 0)  # near "Show all languages" area
]
for y, _ in heading_lines:
    # tiny faint underline to give structure; very light and short so it won't conflict with pasted labels
    draw.line([(heading_x, y + 60), (heading_x + heading_width, y + 60)], fill=muted_line, width=1)

# Done: Background, status bar, header divider, section separators, sort-area backdrop, bottom divider/backdrop.
# The detected UI elements (icons, texts, buttons) will be pasted on top of these shapes.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/19_icon_4.45.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["4.45"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 100, 70)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/21_icon_4.45.png
try:
    _c21 = get_crop(21, 61, 65)
    canvas.paste(_c21, (179, 1), _c21)
except Exception:
    pass
layout["4.45"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 66, 62)
    canvas.paste(_c22, (307, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 52, 69)
    canvas.paste(_c23, (1320, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/24_icon_4.45.png
try:
    _c24 = get_crop(24, 65, 66)
    canvas.paste(_c24, (111, 1), _c24)
except Exception:
    pass
layout["4.45"] = [111, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 52, 63)
    canvas.paste(_c25, (248, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/26_icon_Toggle_to_filter_only_free_events.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 1729), _c26)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/27_icon_Clear_all.png
try:
    _c27 = get_crop(27, 178, 144)
    canvas.paste(_c27, (1214, 72), _c27)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_10_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-12/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
