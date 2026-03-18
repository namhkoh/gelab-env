# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_06
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8.png
# step_index: 6/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background base
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (top)
status_h = 92
draw.rectangle((0, 0, 1440, status_h), fill="#c7c7c7")

# thin darker line under status bar to separate it from header
draw.line((0, status_h, 1440, status_h), fill="#b0b0b0", width=2)

# Header / toolbar area (below status bar)
header_top = status_h
header_h = 116
header_bottom = header_top + header_h
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")

# subtle purple divider under header (to echo app accent without drawing text/icons)
draw.line((24, header_bottom, 1416, header_bottom), fill="#efe7f6", width=6)
draw.line((24, header_bottom+6, 1416, header_bottom+6), fill="#f7f5f9", width=1)

# List area separators (between each selectable row)
# positions derived from detected element boxes; draw soft dividers only
separators_y = [378, 558, 738, 918, 1098, 1278]
for y in separators_y:
    draw.line((48, y, 1392, y), fill="#f3f3f3", width=2)

# subtle left inset guide line for alignment of list items (non-intrusive)
draw.line((48, header_bottom + 36, 48, 1400), fill="#fafafa", width=4)

# faint bottom shadow under the top region to give depth
for i, alpha in enumerate([0x09, 0x06, 0x03], start=0):
    yy = header_bottom + i
    shade = 200 - i*8
    shade_hex = "#{:02x}{:02x}{:02x}".format(shade, shade, shade)
    draw.line((0, yy, 1440, yy), fill=shade_hex, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/00_icon_9.31.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (12, 72), _c0)
except Exception:
    pass
layout["9.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/01_icon_9.31.png
try:
    _c1 = get_crop(1, 62, 63)
    canvas.paste(_c1, (177, 2), _c1)
except Exception:
    pass
layout["9.31"] = [177, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 1344, 144)
    canvas.paste(_c2, (48, 234), _c2)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/03_icon_9.31.png
try:
    _c3 = get_crop(3, 55, 64)
    canvas.paste(_c3, (114, 2), _c3)
except Exception:
    pass
layout["9.31"] = [114, 2, 169, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 54, 59)
    canvas.paste(_c4, (315, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 5, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 55, 61)
    canvas.paste(_c5, (245, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [245, 4, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 61)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1312, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 61)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 123, 129)
    canvas.paste(_c8, (1291, 246), _c8)
except Exception:
    pass
layout["icon_8"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 47, 61)
    canvas.paste(_c9, (384, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [384, 3, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/11_text_9.31.png
try:
    _c11 = get_crop(11, 89, 45)
    canvas.paste(_c11, (20, 15), _c11)
except Exception:
    pass
layout["9.31"] = [20, 15, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_06_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-8/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
