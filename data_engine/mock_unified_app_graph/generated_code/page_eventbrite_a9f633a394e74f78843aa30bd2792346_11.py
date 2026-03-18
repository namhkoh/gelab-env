# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_11
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13.png
# step_index: 11/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page
# Assumes: canvas (1440x2960 PIL Image) and draw (ImageDraw) are provided

# Fill overall background (dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at top (~50-90px). Use a neutral light gray as seen in screenshot.
STATUS_H = 88
draw.rectangle([(0, 0), (1440, STATUS_H)], fill="#D8D8D8")

# Subtle divider under status bar to separate it from toolbar/header area
draw.line([(0, STATUS_H), (1440, STATUS_H)], fill="#E8E8E8", width=1)

# Header/toolbar area (below status bar). Keep background white but add a faint bottom divider/shadow.
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 176
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill="#FFFFFF")
# faint shadow line
draw.line([(0, HEADER_BOTTOM), (1440, HEADER_BOTTOM)], fill="#F0EEF2", width=2)

# Main content area remains white (no content drawn here because text/icons are auto-pasted)
# But draw a very subtle vertical guideline grid to imply structure (very faint, non-intrusive)
grid_color = "#FBFBFB"
for gx in range(48, 1440, 144):  # very faint column guides
    draw.line([(gx, HEADER_BOTTOM + 8), (gx, 2760)], fill=grid_color, width=1)

# Separator line that visually separates content from bottom action area (placed above the button)
SEPARATOR_Y = 2720
draw.line([(48, SEPARATOR_Y), (1392, SEPARATOR_Y)], fill="#F1EFF3", width=2)

# Soft shadow/background behind the bottom "Apply date range" area (do not draw button contents)
# The detected button will be pasted at (48,2768)-(1392,2912); create a subtle larger background behind it.
button_bg_rect = [36, 2748, 1404, 2916]  # slightly larger than the button area
draw.rounded_rectangle(button_bg_rect, radius=12, fill="#FBFBFC", outline="#E4E0E6", width=1)

# Additional subtle left alignment guide block behind the date fields (keeps layout balance)
# This is a faint rounded panel behind the date region; the text will be pasted on top.
panel_rect = [32, 264, 1408, 680]
draw.rounded_rectangle(panel_rect, radius=8, fill="#FFFFFF", outline=None)

# Very faint inner divider lines to suggest grouping within the date area
draw.line([(48, 400), (1392, 400)], fill="#FAF9FB", width=1)
draw.line([(48, 520), (1392, 520)], fill="#FAF9FB", width=1)

# Final subtle edge lines around the canvas to frame the UI
draw.line([(0, 0), (1440, 0)], fill="#DCDCDC", width=1)
draw.line([(0, 2959), (1440, 2959)], fill="#DCDCDC", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/01_icon_4.51.png
try:
    _c1 = get_crop(1, 58, 65)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["4.51"] = [114, 2, 172, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/02_icon_4.51.png
try:
    _c2 = get_crop(2, 57, 63)
    canvas.paste(_c2, (182, 2), _c2)
except Exception:
    pass
layout["4.51"] = [182, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 60)
    canvas.paste(_c3, (310, 4), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 4, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 48, 70)
    canvas.paste(_c4, (1155, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 0, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/06_icon_4.51.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 96, 70)
    canvas.paste(_c7, (1211, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1211, 0, 1307, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 66)
    canvas.paste(_c8, (1325, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/09_icon_4.51.png
try:
    _c9 = get_crop(9, 90, 62)
    canvas.paste(_c9, (16, 3), _c9)
except Exception:
    pass
layout["4.51"] = [16, 3, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/10_icon_What_date.png
try:
    _c10 = get_crop(10, 318, 72)
    canvas.paste(_c10, (558, 112), _c10)
except Exception:
    pass
layout["What_date?"] = [558, 112, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 63)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 589, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/13_text_End_Date.png
try:
    _c13 = get_crop(13, 253, 67)
    canvas.paste(_c13, (45, 437), _c13)
except Exception:
    pass
layout["End_Date"] = [45, 437, 298, 504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_11_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-13/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
