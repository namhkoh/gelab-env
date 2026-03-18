# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_07
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9.png
# step_index: 7/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL Image 1440x2960 RGB, draw: ImageDraw)
# fonts available: font_sm, font_md, font_lg, font_xl

# Overall background (very light off-white to match the screenshot)
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFD")

# Status bar area at top (~88px) - muted gray
status_bar_h = 88
draw.rectangle((0, 0, 1440, status_bar_h), fill="#C7C7C7")

# Subtle darker divider under status bar
draw.line((0, status_bar_h, 1440, status_bar_h), fill="#B7B7B7", width=1)

# Main toolbar area (just under status bar) keep it white but add a very faint bottom divider
toolbar_top = status_bar_h
toolbar_bottom = 200
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill="#FBFBFD")
draw.line((24, toolbar_bottom, 1440-24, toolbar_bottom), fill="#EFEFF4", width=1)

# Prominent underline for the search/input area (purple accent)
# Positioned roughly under the large "Find events in..." input area
underline_y = 392
draw.line((48, underline_y, 1440-48, underline_y), fill="#4B47FF", width=4)

# Light horizontal separators for structure beneath the input area
draw.line((24, 440, 1440-24, 440), fill="#F0F0F5", width=1)
draw.line((24, 640, 1440-24, 640), fill="#F0F0F5", width=1)

# Group card background for the "Nearby / Online events" area
card_left = 40
card_top = 420
card_right = 1400
card_bottom = 600
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=20, fill="#FFFFFF", outline="#EEF0FA", width=1)

# Sub-card separators inside that card (to visually group items without drawing text/icons)
# two thin lines to indicate grouping positions
draw.line((card_left+24, card_top+120, card_right-24, card_top+120), fill="#F4F6FA", width=1)

# Subtle highlight pill behind potential selectable area lower on the screen (background only)
pill_left = 40
pill_top = 740
pill_right = 660
pill_bottom = 860
draw.rounded_rectangle((pill_left, pill_top, pill_right, pill_bottom),
                       radius=14, fill="#FFFFFF", outline="#F2F3F8", width=1)

# Faint divider above the "Browsing in" section
draw.line((24, 720, 1440-24, 720), fill="#F1F2F7", width=1)

# Large empty content area background hint (keeps bottom visually distinct)
content_band_top = 980
content_band_bottom = 1400
draw.rectangle((0, content_band_top, 1440, content_band_bottom), fill="#FFFFFF")
draw.line((24, content_band_top, 1440-24, content_band_top), fill="#F6F7FB", width=1)

# Footer subtle top border to visually end the page area (very faint)
footer_border_y = 2900
draw.line((24, footer_border_y, 1440-24, footer_border_y), fill="#F3F4F8", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/01_icon_7.18.png
try:
    _c1 = get_crop(1, 59, 62)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["7.18"] = [180, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/02_icon_7.18.png
try:
    _c2 = get_crop(2, 57, 62)
    canvas.paste(_c2, (116, 2), _c2)
except Exception:
    pass
layout["7.18"] = [116, 2, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 61)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 3, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/04_icon_7.18.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.18"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 57)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 64, 63)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1276, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 62)
    canvas.paste(_c8, (1262, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1262, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/09_icon_7.18.png
try:
    _c9 = get_crop(9, 93, 62)
    canvas.paste(_c9, (14, 2), _c9)
except Exception:
    pass
layout["7.18"] = [14, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/11_text_FFind_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_07_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-9/17_text_Los_Angeles.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
