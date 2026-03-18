# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_07
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9.png
# step_index: 7/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top) - subtle neutral grey like in the screenshot
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill=(200, 200, 200))

# Toolbar area (where the back arrow lives). Keep it visually separate with a soft divider.
toolbar_top = status_bar_h
toolbar_bottom = 216
# keep toolbar background white (to avoid covering detected icons/text) but add a soft bottom shadow line
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill=(255, 255, 255))
draw.line((24, toolbar_bottom, 1440 - 24, toolbar_bottom), fill=(235, 235, 235), width=2)

# Content grouping background: subtle off-white rounded card behind the list group
group_left = 24
group_right = 1440 - 24
group_top = 200
group_bottom = 1220
draw.rounded_rectangle((group_left, group_top, group_right, group_bottom),
                       radius=18, fill=(250, 250, 250), outline=(245, 245, 245))

# Light separators between the list sections (positions chosen between detected text blocks)
separator_x1 = 48
separator_x2 = 1440 - 48
separators_y = [
    int((234 + 414) / 2),  # between "Anytime" and "Today"
    int((414 + 594) / 2),  # between "Today" and "Tomorrow"
    int((594 + 774) / 2),  # between "Tomorrow" and "This Week"
    int((774 + 954) / 2),  # between "This Week" and "This Weekend"
    int((954 + 1134) / 2), # between "This Weekend" and "Choose a date"
]
for y in separators_y:
    draw.line((separator_x1, y, separator_x2, y), fill=(242, 242, 242), width=1)

# Subtle left gutter guide (visual structural element, very faint)
gutter_x = 48
draw.line((gutter_x, toolbar_bottom + 8, gutter_x, group_bottom - 8), fill=(245, 245, 245), width=1)

# Bottom area - keep plain white but add faint top divider to separate from possible footer content
footer_top = 2520
draw.line((24, footer_top, 1440 - 24, footer_top), fill=(245, 245, 245), width=1)

# Decorative subtle shadow under the group card to give depth
shadow_top = group_bottom
shadow_bottom = group_bottom + 12
for i, alpha in enumerate([18, 12, 8, 6]):
    y = shadow_top + i
    draw.rectangle((group_left + 2, y, group_right - 2, y + 1), fill=(220, 220, 220, alpha))

# Ensure no text or icons are drawn here (they will be pasted later).
# End of background and structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/00_icon_5.31.png
try:
    _c0 = get_crop(0, 59, 64)
    canvas.paste(_c0, (114, 2), _c0)
except Exception:
    pass
layout["5.31"] = [114, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/01_icon_5.31.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["5.31"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/02_icon_5.31.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (12, 72), _c2)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 61)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 59)
    canvas.paste(_c5, (248, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 4, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 61)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/08_icon_5.31.png
try:
    _c8 = get_crop(8, 89, 62)
    canvas.paste(_c8, (16, 3), _c8)
except Exception:
    pass
layout["5.31"] = [16, 3, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_07_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-9/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
