# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_03
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5.png
# step_index: 3/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the provided canvas/draw
# Available objects: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm,font_md,font_lg,font_xl

# Colors
bg_color = (250, 250, 252)         # very light off-white background
status_bar_color = (189, 189, 189) # muted grey for status bar
status_border = (158, 158, 158)    # darker border under status bar
accent_blue = (57, 73, 255)        # bright accent blue (underline)
pale_accent = (231, 242, 255)      # pale blue for circular icon backgrounds
section_bg = (247, 248, 251)       # very subtle section background
divider = (236, 237, 240)          # faint divider lines

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)
draw.line([(0, status_h), (W, status_h)], fill=status_border, width=1)

# Header area (below status) - keep it visually grouped but not drawing any text/icons
header_top = status_h
header_bottom = 420
# subtle white (same as bg) area - we draw a faint shadow strip to separate from content
draw.rectangle([(0, header_top), (W, header_bottom)], fill=bg_color)
draw.line([(48, header_bottom), (W-48, header_bottom)], fill=divider, width=1)

# Big accent underline for the search field (structural element)
underline_y = 396
draw.line([(48, underline_y), (W-48, underline_y)], fill=accent_blue, width=5)

# Circular pale backgrounds for the two option icons (Nearby / Online events).
# These are just background shapes; the actual icons/text will be pasted on top.
left_circle_center = (140, 420)
right_circle_center = (640, 420)
circle_r = 54
draw.ellipse([
    (left_circle_center[0]-circle_r, left_circle_center[1]-circle_r),
    (left_circle_center[0]+circle_r, left_circle_center[1]+circle_r)
], fill=pale_accent)
draw.ellipse([
    (right_circle_center[0]-circle_r, right_circle_center[1]-circle_r),
    (right_circle_center[0]+circle_r, right_circle_center[1]+circle_r)
], fill=pale_accent)

# Faint divider below the options area
options_div_y = 520
draw.line([(48, options_div_y), (W-48, options_div_y)], fill=divider, width=1)

# Subtle rounded panel background for the "Browsing in" / city selection area
panel_top = 700
panel_bottom = 980
panel_margin_x = 36
radius = 18
# Pillow's ImageDraw in modern versions supports rounded_rectangle; use it if available.
try:
    draw.rounded_rectangle(
        [(panel_margin_x, panel_top), (W-panel_margin_x, panel_bottom)],
        radius=radius, fill=section_bg, outline=None
    )
except Exception:
    # Fallback: draw normal rectangle if rounded not available
    draw.rectangle([(panel_margin_x, panel_top), (W-panel_margin_x, panel_bottom)], fill=section_bg)

# Very light horizontal separator near where the city name block ends
draw.line([(panel_margin_x+8, panel_bottom), (W-panel_margin_x-8, panel_bottom)], fill=divider, width=1)

# A subtle left-side vertical guide line to help visually separate content columns (very faint)
draw.line([(48, panel_top+12), (48, H-48)], fill=(245,245,247), width=1)

# End of structural/background painting.
# (Do not draw any icons or text — those will be pasted on top by the pipeline.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/01_icon_8.11.png
try:
    _c1 = get_crop(1, 59, 65)
    canvas.paste(_c1, (113, 1), _c1)
except Exception:
    pass
layout["8.11"] = [113, 1, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/02_icon_8.11.png
try:
    _c2 = get_crop(2, 59, 62)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["8.11"] = [180, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 60)
    canvas.paste(_c3, (310, 4), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 4, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/04_icon_8.11.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["8.11"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 59)
    canvas.paste(_c6, (1322, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 62, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1275, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 62)
    canvas.paste(_c8, (1265, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1265, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/10_icon_8.11.png
try:
    _c10 = get_crop(10, 91, 62)
    canvas.paste(_c10, (14, 2), _c10)
except Exception:
    pass
layout["8.11"] = [14, 2, 105, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/11_text_FFind_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_03_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-5/17_text_Chicago.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 816, 1440, 954]
