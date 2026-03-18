# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_03
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-5.png
# step_index: 3/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas is provided)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar background (~72px high)
status_bar_h = 72
status_bar_color = (189, 189, 189)  # light grey
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=status_bar_color)

# Subtle divider under the status bar
draw.line([(0, status_bar_h), (1440, status_bar_h)], fill=(160, 160, 160), width=1)

# Header bottom divider (separates header/search area from content)
header_div_y = 170
draw.line([(48, header_div_y), (1392, header_div_y)], fill=(235, 235, 235), width=1)

# Search underline (accent indigo line under the search field)
search_line_y = 393
accent_indigo = (63, 81, 181)  # indigo accent
draw.line([(48, search_line_y), (1392, search_line_y)], fill=accent_indigo, width=4)

# Light glow/soft shadow under search line (subtle)
draw.line([(48, search_line_y + 6), (1392, search_line_y + 6)], fill=(245, 245, 250), width=1)

# Options card background (behind the "Nearby" / "Online events" option group)
options_card_bbox = (36, 420, 1404, 600)
options_card_fill = (247, 249, 255)  # very light bluish-white
options_card_outline = (225, 226, 235)
draw.rounded_rectangle(options_card_bbox, radius=20, fill=options_card_fill, outline=options_card_outline, width=1)

# Subtle separator below the options area
options_sep_y = 620
draw.line([(36, options_sep_y), (1404, options_sep_y)], fill=(240, 240, 245), width=1)

# Browsing/location card background (rounded card behind "Browsing in / San Francisco")
browse_card_bbox = (36, 720, 1404, 940)
browse_card_fill = (255, 255, 255)  # keep it white but with soft border to read as a card
browse_card_outline = (240, 240, 245)
draw.rounded_rectangle(browse_card_bbox, radius=18, fill=browse_card_fill, outline=browse_card_outline, width=1)

# Very faint vertical guide line (purely structural, subtle) on left margin to anchor content
draw.line([(48, status_bar_h + 20), (48, 2960 - 40)], fill=(250, 250, 250), width=2)

# Bottom large area: keep clear but add a very faint horizontal baseline to imply continuation
bottom_baseline_y = 1000
draw.line([(36, bottom_baseline_y), (1404, bottom_baseline_y)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/01_icon_7.28.png
try:
    _c1 = get_crop(1, 60, 63)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["7.28"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/02_icon_7.28.png
try:
    _c2 = get_crop(2, 59, 64)
    canvas.paste(_c2, (115, 1), _c2)
except Exception:
    pass
layout["7.28"] = [115, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 62, 62)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/04_icon_7.28.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.28"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 59)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 78, 62)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 1, 1291, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 62)
    canvas.paste(_c8, (1264, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1264, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/09_icon_7.28.png
try:
    _c9 = get_crop(9, 92, 63)
    canvas.paste(_c9, (15, 1), _c9)
except Exception:
    pass
layout["7.28"] = [15, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/11_text_FFind_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/17_text_San_Francisco.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["San_Francisco"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_03_2024_4_23_19_27_45f56b06f31541079045047b6d542613-5/18_text_California.png
try:
    _c18 = get_crop(18, 188, 50)
    canvas.paste(_c18, (42, 902), _c18)
except Exception:
    pass
layout["California"] = [42, 902, 230, 952]
