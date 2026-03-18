# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_04
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6.png
# step_index: 4/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas/draw context.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background with a very light off-white (dominant color)
draw.rectangle((0, 0, 1440, 2960), fill="#FBFDFF")

# Top status bar (dark grey area for system icons)
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill="#7A7A7A")

# Thin separator between status bar and header
draw.line((0, status_bar_h, 1440, status_bar_h), fill="#6d6d6d", width=1)

# Header / toolbar area (below the status bar)
toolbar_top = status_bar_h
toolbar_bottom = 200
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill="#FFFFFF")

# Subtle shadow line under toolbar
draw.line((0, toolbar_bottom-1, 1440, toolbar_bottom-1), fill="#E8E9EB", width=1)

# Prominent blue underline (search field indicator) across the header area
underline_y = toolbar_bottom - 8
draw.line((48, underline_y, 1440-48, underline_y), fill="#1A57FF", width=4)

# Section cards / background areas for the selectable location options (rounded rectangles).
# Left card behind "Nearby" (slightly larger than detected content to act as background)
left_card = (36, 420, 483, 599)
draw.rounded_rectangle(left_card, radius=24, fill="#F2F6FF", outline=None)

# Right card behind "Online events" (slightly larger than detected content)
right_card = (500, 420, 983, 599)
draw.rounded_rectangle(right_card, radius=24, fill="#F2F6FF", outline=None)

# Subtle boundaries / separators below the option cards
separator_y = 620
draw.line((36, separator_y, 1440-36, separator_y), fill="#ECEFF3", width=1)

# Large faint content area band to suggest content region (keeps space for pasted content/icons)
content_band_top = separator_y + 24
content_band_bottom = content_band_top + 1100
draw.rectangle((24, content_band_top, 1440-24, content_band_bottom), fill="#FFFFFF", outline="#F5F7FA")

# Very light vertical guide lines to subtly structure content columns (non-intrusive)
draw.line((360, content_band_top, 360, content_band_bottom), fill="#FAFBFD", width=1)
draw.line((720, content_band_top, 720, content_band_bottom), fill="#FAFBFD", width=1)
draw.line((1080, content_band_top, 1080, content_band_bottom), fill="#FAFBFD", width=1)

# Footer thin divider near bottom of the content band
draw.line((24, content_band_bottom, 1440-24, content_band_bottom), fill="#ECEFF3", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 96, 67)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1310, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 60, 62)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/03_icon_Loading.png
try:
    _c3 = get_crop(3, 364, 418)
    canvas.paste(_c3, (543, 1627), _c3)
except Exception:
    pass
layout["Loading"] = [543, 1627, 907, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/04_icon_7.34.png
try:
    _c4 = get_crop(4, 59, 65)
    canvas.paste(_c4, (114, 1), _c4)
except Exception:
    pass
layout["7.34"] = [114, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/05_icon_7.34.png
try:
    _c5 = get_crop(5, 61, 65)
    canvas.paste(_c5, (179, 1), _c5)
except Exception:
    pass
layout["7.34"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/06_icon_7.34.png
try:
    _c6 = get_crop(6, 168, 168)
    canvas.paste(_c6, (0, 72), _c6)
except Exception:
    pass
layout["7.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 58)
    canvas.paste(_c7, (251, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [251, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 81, 93)
    canvas.paste(_c8, (1313, 287), _c8)
except Exception:
    pass
layout["icon_8"] = [1313, 287, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 65)
    canvas.paste(_c9, (1320, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1320, 0, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/10_icon_7.34.png
try:
    _c10 = get_crop(10, 92, 65)
    canvas.paste(_c10, (16, 0), _c10)
except Exception:
    pass
layout["7.34"] = [16, 0, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 50, 66)
    canvas.paste(_c12, (383, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/13_text_Los_Angeles.png
try:
    _c13 = get_crop(13, 1344, 129)
    canvas.paste(_c13, (48, 264), _c13)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/14_text_Online_events.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_04_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]
