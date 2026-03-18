# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_03
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5.png
# step_index: 3/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (ensure clean white)
draw.rectangle([(0, 0), (canvas.width, canvas.height)], fill=(255, 255, 255))

# Status bar (top ~72px)
status_h = 72
status_color = (190, 190, 190)  # light gray status bar background
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# thin darker divider under status bar
draw.line([(0, status_h), (canvas.width, status_h)], fill=(170, 170, 170), width=2)

# Header area (toolbar) under status bar
header_top = status_h
header_h = 88
header_bottom = header_top + header_h
# keep header background white but add a very subtle shadow line beneath
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill=(255, 255, 255))
draw.line([(24, header_bottom), (canvas.width - 24, header_bottom)], fill=(230, 226, 238), width=3)

# Main content container (rounded rectangle) behind the list items
container_margin_x = 24
container_top = 200
container_bottom = 1320
container_bbox = [container_margin_x, container_top, canvas.width - container_margin_x, container_bottom]
# subtle off-white card with light border
card_fill = (255, 255, 255)
card_outline = (238, 238, 245)
try:
    draw.rounded_rectangle(container_bbox, radius=18, fill=card_fill, outline=card_outline, width=1)
except AttributeError:
    # Fallback: regular rectangle if rounded not available
    draw.rectangle(container_bbox, fill=card_fill, outline=card_outline)

# Inner separators between options (use detected item vertical positions to align)
# Detected item tops: 234, 414, 594, 774, 954, 1134 ; heights 144 each -> separators at top+height
item_tops = [234, 414, 594, 774, 954, 1134]
sep_color = (245, 244, 247)  # very light divider
sep_x1 = container_margin_x + 24
sep_x2 = canvas.width - container_margin_x - 24
for t in item_tops:
    sep_y = t + 144  # bottom edge of each item block
    # only draw separators that lie within container
    if container_top < sep_y < container_bottom:
        draw.line([(sep_x1, sep_y), (sep_x2, sep_y)], fill=sep_color, width=2)

# Slight left inset rule to visually group items (subtle vertical guide)
guide_x = container_margin_x + 18
draw.line([(guide_x, container_top + 12), (guide_x, container_bottom - 12)], fill=(250, 250, 251), width=1)

# Top-left subtle back-area highlight (background only, icon/text will be pasted on top)
back_area_bbox = [24, header_top + 12, 96, header_bottom - 12]
draw.rectangle(back_area_bbox, fill=(255, 255, 255))

# Right-side small hit-area background near top-right (for check/controls) — background only
right_area_bbox = [canvas.width - 120, header_top + 20, canvas.width - 24, header_bottom - 20]
draw.rectangle(right_area_bbox, fill=(255, 255, 255))

# Final subtle bottom shadow under container to separate from page (soft line)
shadow_y = container_bottom + 8
if shadow_y < canvas.height:
    draw.line([(container_margin_x + 8, shadow_y), (canvas.width - container_margin_x - 8, shadow_y)],
              fill=(245, 245, 247), width=3)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/00_icon_7.57.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["7.57"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/01_icon_7.57.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["7.57"] = [114, 2, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 64, 61)
    canvas.paste(_c2, (308, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/03_icon_7.57.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["7.57"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 44, 60)
    canvas.paste(_c6, (1326, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 100, 61)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/08_icon_7.57.png
try:
    _c8 = get_crop(8, 90, 62)
    canvas.paste(_c8, (17, 3), _c8)
except Exception:
    pass
layout["7.57"] = [17, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_03_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-5/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
