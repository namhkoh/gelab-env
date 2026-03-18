# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_04
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6.png
# step_index: 4/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structured background for Eventbrite "San Francisco" page

# Colors
STATUS_BAR_COLOR = (189, 189, 189)   # light gray status bar
HEADER_UNDERLINE = (45, 88, 255)     # vivid blue underline
SECTION_BORDER = (230, 230, 235)     # very light neutral border for section cards
SECTION_BG = (249, 250, 252)         # extremely light bluish/gray background
SEPARATOR = (240, 240, 245)          # subtle separator line

w, h = canvas.size

# 1) Status bar area at top (~72px high)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BAR_COLOR)

# 2) Header area (below status bar) - keep it visually elevated but mostly white
header_top = status_h
header_bottom = 330
# White background (canvas is white already) but draw a very subtle shadow line to separate
draw.rectangle([(0, header_top), (w, header_bottom)], fill=(255, 255, 255))
# subtle top divider (to separate from status)
draw.line([(0, header_top), (w, header_top)], fill=(225,225,225), width=1)

# 3) Blue underline under the header title (full width with side margins)
underline_left = 48
underline_right = w - 48
underline_y = header_bottom - 2
draw.rectangle([(underline_left, underline_y), (underline_right, underline_y + 4)], fill=HEADER_UNDERLINE)

# 4) Two section card backgrounds (rounded rectangles) behind the option groups
# Left card (e.g., Nearby / Current location area)
left_card = (48, 390, 480, 560)
draw.rounded_rectangle(left_card, radius=20, fill=SECTION_BG, outline=SECTION_BORDER, width=1)

# Right card (e.g., Online events / Virtual attendance area)
right_card = (510, 390, 966, 560)
draw.rounded_rectangle(right_card, radius=20, fill=SECTION_BG, outline=SECTION_BORDER, width=1)

# 5) Thin subtle separator line under the option area
sep_y = 600
draw.line([(48, sep_y), (w - 48, sep_y)], fill=SEPARATOR, width=1)

# 6) Large faint content area hint (keeps main canvas visually grouped)
content_top = sep_y + 24
content_bottom = 1820
content_margin = 48
draw.rounded_rectangle((content_margin, content_top, w - content_margin, content_bottom),
                       radius=12, outline=SECTION_BORDER, width=1, fill=(255,255,255))

# 7) Additional faint center marker for loading region (a light tiny circle background)
# (This is only a subtle background hint, not text/icon)
loading_center_x = w // 2
loading_center_y = 1970
draw.ellipse([(loading_center_x - 6, loading_center_y - 6), (loading_center_x + 6, loading_center_y + 6)],
             fill=(245,245,248))

# 8) Final bottom separator to anchor the page visually
draw.line([(0, h - 1), (w, h - 1)], fill=(230,230,230), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 61, 62)
    canvas.paste(_c1, (310, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/02_icon_8.11.png
try:
    _c2 = get_crop(2, 168, 168)
    canvas.paste(_c2, (0, 72), _c2)
except Exception:
    pass
layout["8.11"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 95, 66)
    canvas.paste(_c3, (1215, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1215, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/04_icon_8.11.png
try:
    _c4 = get_crop(4, 61, 65)
    canvas.paste(_c4, (113, 1), _c4)
except Exception:
    pass
layout["8.11"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/05_icon_8.11.png
try:
    _c5 = get_crop(5, 61, 64)
    canvas.paste(_c5, (179, 1), _c5)
except Exception:
    pass
layout["8.11"] = [179, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 49, 57)
    canvas.paste(_c6, (249, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 81, 92)
    canvas.paste(_c7, (1313, 288), _c7)
except Exception:
    pass
layout["icon_7"] = [1313, 288, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 66)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/10_icon_8.11.png
try:
    _c10 = get_crop(10, 98, 66)
    canvas.paste(_c10, (11, 0), _c10)
except Exception:
    pass
layout["8.11"] = [11, 0, 109, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/11_text_San_Francisco.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["San_Francisco"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_04_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-6/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
