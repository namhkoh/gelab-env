# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_13
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-15.png
# step_index: 13/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill base background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top)
status_h = 86
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")
# subtle bottom border of status bar
draw.rectangle([(0, status_h - 2), (1440, status_h)], fill="#BDBDBD")

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# thin accent underline for header
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#2D2242", width=3)
draw.line([(0, header_bottom + 2), (1440, header_bottom + 2)], fill="#F0F0F0", width=1)

# Section card backgrounds (rounded rectangles behind each large text row)
section_x0 = 48
section_x1 = 1392
section_rows = [234, 414, 594, 774, 954, 1134]
for idx, y in enumerate(section_rows):
    top = y - 12
    bottom = y + 144 + 12
    # Slight highlight for the first (selected) row, subtle white for others
    if idx == 0:
        fill_color = "#FFF5EB"   # very light orange behind the selected item
        outline_color = "#FFD1A6"
    else:
        fill_color = "#FFFFFF"
        outline_color = None
    draw.rounded_rectangle([(section_x0 - 6, top), (section_x1 + 6, bottom)], radius=18,
                           fill=fill_color, outline=outline_color, width=2 if outline_color else 0)

# Thin separators between sections
for y in [r + 144 + 6 for r in section_rows[:-1]]:  # separators after each row except last
    sep_y = y
    draw.line([(section_x0, sep_y), (section_x1, sep_y)], fill="#E9E9E9", width=1)

# Large content area background below the list
content_top = 1400
draw.rectangle([(0, content_top), (1440, 2960)], fill="#FAFAFA")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/00_icon_7.29.png
try:
    _c0 = get_crop(0, 58, 61)
    canvas.paste(_c0, (181, 3), _c0)
except Exception:
    pass
layout["7.29"] = [181, 3, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 64, 60)
    canvas.paste(_c1, (308, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [308, 4, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/02_icon_7.29.png
try:
    _c2 = get_crop(2, 57, 62)
    canvas.paste(_c2, (115, 3), _c2)
except Exception:
    pass
layout["7.29"] = [115, 3, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/03_icon_7.29.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 63)
    canvas.paste(_c5, (1154, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 4, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 99, 62)
    canvas.paste(_c6, (1215, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1215, 2, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 58)
    canvas.paste(_c7, (248, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 60)
    canvas.paste(_c8, (1325, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/09_icon_7.29.png
try:
    _c9 = get_crop(9, 90, 60)
    canvas.paste(_c9, (17, 4), _c9)
except Exception:
    pass
layout["7.29"] = [17, 4, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 123, 128)
    canvas.paste(_c10, (1291, 247), _c10)
except Exception:
    pass
layout["icon_10"] = [1291, 247, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/11_icon_Tomorrow.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 594), _c11)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_13_2024_4_23_19_27_45f56b06f31541079045047b6d542613-15/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
