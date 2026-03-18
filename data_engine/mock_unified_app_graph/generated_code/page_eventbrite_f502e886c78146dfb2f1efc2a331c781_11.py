# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_11
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13.png
# step_index: 11/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw general background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar (top) area
status_h = 110
draw.rectangle([(0, 0), (1440, status_h)], fill="#D0D0D0")

# Thin divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#C2C2C2", width=2)

# Top toolbar area (below status bar) with subtle shadow divider
toolbar_top = status_h
toolbar_bottom = 200
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")
draw.line([(0, toolbar_bottom), (1440, toolbar_bottom)], fill="#E9E9E9", width=2)

# Soft rounded container behind the list region (subtle outline only)
# Use margins to avoid drawing over detected icons/text (they will be pasted on top)
container_left = 36
container_right = 1440 - 36
text_positions = [234, 414, 594, 774, 954, 1134]  # detected text y positions
text_h = 144
container_top = min(text_positions) - 28
container_bottom = max(text_positions) + text_h + 28
draw.rounded_rectangle(
    [(container_left, container_top), (container_right, container_bottom)],
    radius=14,
    outline="#F6F6F6",
    width=1,
    fill=None
)

# Draw subtle separators below each text block (light, thin lines)
for y in text_positions:
    sep_y = y + text_h
    # keep separators inset from the very edges to avoid overlapping potential icons
    inset = 48
    draw.line([(inset, sep_y), (1440 - inset, sep_y)], fill="#F4F4F4", width=2)
    # tiny highlight above the separator for depth
    draw.line([(inset, sep_y - 1), (1440 - inset, sep_y - 1)], fill="#FFFFFF", width=1)

# Light bottom band to anchor the page visually
bottom_band_top = container_bottom + 22
draw.rectangle([(0, bottom_band_top), (1440, bottom_band_top + 6)], fill="#F7F7F7")

# Slight vignette shading near top edges to match screenshot feel (very subtle)
for i, alpha in enumerate([6, 5, 4, 3, 2, 1]):
    y = toolbar_top + i * 2
    draw.line([(0, y), (1440, y)], fill=(220, 220, 220, alpha))

# End - leave all icons/text to be pasted on top at detected positions

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/00_icon_7.19.png
try:
    _c0 = get_crop(0, 59, 62)
    canvas.paste(_c0, (181, 2), _c0)
except Exception:
    pass
layout["7.19"] = [181, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 63, 60)
    canvas.paste(_c1, (309, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [309, 4, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/02_icon_7.19.png
try:
    _c2 = get_crop(2, 56, 63)
    canvas.paste(_c2, (116, 3), _c2)
except Exception:
    pass
layout["7.19"] = [116, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/03_icon_7.19.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 63)
    canvas.paste(_c5, (1154, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 4, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 98, 62)
    canvas.paste(_c6, (1216, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1216, 2, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 60)
    canvas.paste(_c7, (1325, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [1325, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 59)
    canvas.paste(_c8, (248, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [248, 4, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/10_icon_7.19.png
try:
    _c10 = get_crop(10, 92, 62)
    canvas.paste(_c10, (16, 3), _c10)
except Exception:
    pass
layout["7.19"] = [16, 3, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/11_icon_Tomorrow.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 594), _c11)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_11_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-13/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
