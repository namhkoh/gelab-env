# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_14
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-16.png
# step_index: 14/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill background (dominant color: white)
draw.rectangle([(0, 0), (canvas.width, canvas.height)], fill=(255, 255, 255))

# Status bar area at top (~72px) - light gray
status_bar_height = 72
status_bar_color = (191, 191, 191)  # light gray
draw.rectangle([(0, 0), (canvas.width, status_bar_height)], fill=status_bar_color)

# Header / toolbar area below status bar (subtle white on white to provide structure)
header_top = status_bar_height
header_height = 120
header_color = (255, 255, 255)  # keep header visually consistent with background
draw.rectangle([(0, header_top), (canvas.width, header_top + header_height)], fill=header_color)

# Divider line under header
divider_y = header_top + header_height
draw.line([(24, divider_y), (canvas.width - 24, divider_y)], fill=(230, 230, 230), width=1)

# Subtle section separator under the date block area (keeps content areas distinct)
# Place it well above the interactive "Choose a date" clickable area to avoid overlap
section_separator_y = 520
draw.line([(48, section_separator_y), (canvas.width - 48, section_separator_y)], fill=(245, 245, 245), width=1)

# Light shadow band above the bottom action area (do not draw the button itself)
# Button top is at y=2768 (auto-pasted), so draw a faint band above it
shadow_band_top = 2728
shadow_band_bottom = 2764
shadow_color = (245, 245, 247)
draw.rectangle([(24, shadow_band_top), (canvas.width - 24, shadow_band_bottom)], fill=shadow_color)

# Very thin top stroke to accentuate separation from content to bottom CTA
draw.line([(24, shadow_band_top), (canvas.width - 24, shadow_band_top)], fill=(220, 220, 225), width=1)

# Optional: subtle left/right margins visual guide (very faint) to mimic UI gutters
gutters_color = (250, 250, 250)
gutters_width = 1
draw.line([(48, 0), (48, canvas.height)], fill=gutters_color, width=gutters_width)
draw.line([(canvas.width - 48, 0), (canvas.width - 48, canvas.height)], fill=gutters_color, width=gutters_width)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/01_icon_7.29.png
try:
    _c1 = get_crop(1, 56, 62)
    canvas.paste(_c1, (183, 2), _c1)
except Exception:
    pass
layout["7.29"] = [183, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 60)
    canvas.paste(_c2, (310, 4), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 4, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/03_icon_7.29.png
try:
    _c3 = get_crop(3, 57, 63)
    canvas.paste(_c3, (115, 3), _c3)
except Exception:
    pass
layout["7.29"] = [115, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 48, 70)
    canvas.paste(_c4, (1155, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 0, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/06_icon_7.29.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 95, 70)
    canvas.paste(_c7, (1211, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1211, 0, 1306, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 66)
    canvas.paste(_c8, (1325, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/09_icon_7.29.png
try:
    _c9 = get_crop(9, 89, 61)
    canvas.paste(_c9, (17, 3), _c9)
except Exception:
    pass
layout["7.29"] = [17, 3, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/10_icon_What_date.png
try:
    _c10 = get_crop(10, 318, 73)
    canvas.paste(_c10, (558, 111), _c10)
except Exception:
    pass
layout["What_date?"] = [558, 111, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 63)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 580, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/13_text_End_Date.png
try:
    _c13 = get_crop(13, 580, 144)
    canvas.paste(_c13, (48, 313), _c13)
except Exception:
    pass
layout["End_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_14_2024_4_23_19_27_45f56b06f31541079045047b6d542613-16/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
