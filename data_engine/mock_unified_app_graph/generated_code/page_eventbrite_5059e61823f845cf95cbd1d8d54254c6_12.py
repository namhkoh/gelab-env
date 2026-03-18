# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_12
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14.png
# step_index: 12/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar background
draw.rectangle([(0, 0), (1440, 72)], fill="#EDEDED")

# Subtle divider under the status bar
draw.line([(0, 72), (1440, 72)], fill="#DAD7DD", width=1)

# Light header shadow band to give separation under the header/title area
draw.rectangle([(0, 160), (1440, 166)], fill="#F3EEF6")

# Very subtle horizontal divider where the main content area starts
draw.line([(48, 220), (1392, 220)], fill="#F0EDF1", width=1)

# Decorative, very faint rounded card shape behind the date selection area (background only)
# Positioned so it does not draw any of the detected text or icons themselves
draw.rounded_rectangle([(36, 260), (1404, 560)], radius=14, fill="#FFFFFF", outline="#F0EBF3", width=1)

# Another faint separator lower in the page to rhythmically break the large white area
draw.line([(48, 920), (1392, 920)], fill="#F5F3F6", width=1)
draw.line([(48, 1540), (1392, 1540)], fill="#F5F3F6", width=1)

# Subtle top shadow for the bottom action area (keeps the detected button free to be pasted on top)
apply_top = 2768
shadow_top = apply_top - 28
draw.rectangle([(36, shadow_top), (1404, shadow_top + 4)], fill="#EDE8EE")

# Very faint border line immediately above the apply button (acts as a separator)
draw.line([(36, apply_top - 2), (1404, apply_top - 2)], fill="#D3CED5", width=1)

# Very faint full-width page edges to suggest device bezel
draw.line([(0, 0), (0, 2960)], fill="#F1EFF2", width=1)
draw.line([(1439, 0), (1439, 2960)], fill="#F1EFF2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/01_icon_7.35.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (182, 2), _c1)
except Exception:
    pass
layout["7.35"] = [182, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 61)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/03_icon_7.35.png
try:
    _c3 = get_crop(3, 57, 63)
    canvas.paste(_c3, (115, 3), _c3)
except Exception:
    pass
layout["7.35"] = [115, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 48, 70)
    canvas.paste(_c4, (1155, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 0, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/06_icon_7.35.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["7.35"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 95, 70)
    canvas.paste(_c7, (1211, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1211, 0, 1306, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 66)
    canvas.paste(_c8, (1325, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/09_icon_What_date.png
try:
    _c9 = get_crop(9, 318, 73)
    canvas.paste(_c9, (558, 111), _c9)
except Exception:
    pass
layout["What_date?"] = [558, 111, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/10_icon_7.35.png
try:
    _c10 = get_crop(10, 91, 62)
    canvas.paste(_c10, (16, 3), _c10)
except Exception:
    pass
layout["7.35"] = [16, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 63)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 580, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/13_text_End_Date.png
try:
    _c13 = get_crop(13, 580, 144)
    canvas.paste(_c13, (48, 313), _c13)
except Exception:
    pass
layout["End_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_12_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-14/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
