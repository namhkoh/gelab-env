# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_08
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10.png
# step_index: 8/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Drawing the background and structural chrome for the UI page
# (Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw) objects.)

# Clear/fill background (dominant is white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at top (~72px) - light gray
status_bar_h = 72
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#CFCFCF")

# Slight overlay to give status bar a subtle darker band at very top (thin)
draw.rectangle([(0, 0), (1440, 6)], fill="#BDBDBD")

# Header area (page title area) - keep it white but give a subtle bottom shadow
header_top = status_bar_h
header_bottom = 264 + 129  # using detected title box to align divider under title
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# Subtle shadow line under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#E6E9F2", width=2)

# Prominent blue underline beneath the title (structure element)
underline_y = header_bottom - 8
draw.line([(48, underline_y), (1440-48, underline_y)], fill="#2B56F5", width=6)

# Section cards / badges for "Nearby" and "Online events"
# Left badge area (use the detected Nearby bbox for placement reference)
left_badge_bbox = (48, 465, 48+415, 465+114)
left_center = (48 + 60, 465 + 57)
right_badge_bbox = (511, 465, 511+415, 465+114)
right_center = (511 + 60, 465 + 57)

# Pale-blue circular backgrounds behind icons (subtle, not icons/text)
badge_radius = 44
draw.ellipse([
    (left_center[0]-badge_radius, left_center[1]-badge_radius),
    (left_center[0]+badge_radius, left_center[1]+badge_radius)
], fill="#E8F4FF")
draw.ellipse([
    (right_center[0]-badge_radius, right_center[1]-badge_radius),
    (right_center[0]+badge_radius, right_center[1]+badge_radius)
], fill="#E8F4FF")

# Add faint inner ring to both badges to match structural chrome
inner_ring_radius = badge_radius - 8
draw.ellipse([
    (left_center[0]-inner_ring_radius, left_center[1]-inner_ring_radius),
    (left_center[0]+inner_ring_radius, left_center[1]+inner_ring_radius)
], outline="#C7E6FF", width=3)
draw.ellipse([
    (right_center[0]-inner_ring_radius, right_center[1]-inner_ring_radius),
    (right_center[0]+inner_ring_radius, right_center[1]+inner_ring_radius)
], outline="#C7E6FF", width=3)

# Row separator under the badges (thin, subtle)
sep_y = left_badge_bbox[1] + left_badge_bbox[3] // 6 + 80
draw.line([(24, sep_y), (1440-24, sep_y)], fill="#F0F1F5", width=1)

# Main content area background (large white card-like area)
content_top = sep_y + 24
content_bottom = 2600
content_left = 24
content_right = 1440 - 24
draw.rounded_rectangle(
    [(content_left, content_top), (content_right, content_bottom)],
    radius=12, fill="#FFFFFF", outline="#F3F4F8", width=1
)

# Subtle dividing bands to indicate sections (structural only)
# 1) A faint band near center to hint loading/content region
band_y = 1400
draw.rectangle([(content_left+8, band_y), (content_right-8, band_y+2)], fill="#F3F4F8")

# 2) Light footer/top-of-list separator
footer_sep_y = 2300
draw.line([(content_left+8, footer_sep_y), (content_right-8, footer_sep_y)], fill="#F0F1F5", width=1)

# Decorative faint corner radii shadow on the main content card (very subtle)
shadow_color = (0, 0, 0, 10)  # semi for conceptual; PIL.ImageDraw will ignore alpha, simulate with very light gray
draw.line([(content_left+12, content_bottom-6), (content_right-12, content_bottom-6)], fill="#F5F6F9", width=2)

# End of structural drawing. Icons, text and interactive elements will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 66)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1311, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 62, 62)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/03_icon_7.19.png
try:
    _c3 = get_crop(3, 61, 65)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["7.19"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/04_icon_7.19.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.19"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/05_icon_7.19.png
try:
    _c5 = get_crop(5, 60, 65)
    canvas.paste(_c5, (115, 1), _c5)
except Exception:
    pass
layout["7.19"] = [115, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 81, 92)
    canvas.paste(_c6, (1313, 288), _c6)
except Exception:
    pass
layout["icon_6"] = [1313, 288, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 58)
    canvas.paste(_c7, (249, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 65)
    canvas.paste(_c8, (1319, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1319, 0, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 67)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/10_icon_Nearby.png
try:
    _c10 = get_crop(10, 415, 114)
    canvas.paste(_c10, (48, 465), _c10)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/11_icon_Loading.png
try:
    _c11 = get_crop(11, 365, 438)
    canvas.paste(_c11, (544, 1606), _c11)
except Exception:
    pass
layout["Loading"] = [544, 1606, 909, 2044]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/12_text_7.19.png
try:
    _c12 = get_crop(12, 91, 45)
    canvas.paste(_c12, (20, 15), _c12)
except Exception:
    pass
layout["7.19"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/13_text_San_Francisco.png
try:
    _c13 = get_crop(13, 1344, 129)
    canvas.paste(_c13, (48, 264), _c13)
except Exception:
    pass
layout["San_Francisco"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/14_text_Online_events.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_08_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-10/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]
