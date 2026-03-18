# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_08
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10.png
# step_index: 8/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background & structural UI elements for the mobile page
# Canvas size: 1440x2960, variables provided: canvas, draw, font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (200, 200, 200)    # light gray status bar
toolbar_divider = (235, 235, 240)     # subtle divider under header
card_shadow = (245, 246, 248)         # very light shadow behind cards
card_bg = (255, 255, 255)             # card/background white
accent_orange = (230, 90, 30)         # accent for selected item
separator = (242, 242, 246)           # thin separators

W, H = canvas.size

# 1) Status bar area at top (~80px tall)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# 2) Header/toolbar area (header sits below status bar)
header_top = status_h
header_h = 96
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill=card_bg)
# thin divider under header
draw.line([(24, header_top + header_h - 1), (W - 24, header_top + header_h - 1)], fill=toolbar_divider, width=1)

# 3) List/section cards backgrounds (rounded rectangles) for detected item positions.
# Detected item bounding boxes (from detection): each at x=48, widths 1344, heights 144 at various y positions.
card_x = 48
card_w = 1344
card_h = 144
card_radius = 20

# Y positions for the list items (taken from detection): approximate top-left Ys
item_ys = [234, 414, 594, 774, 954, 1134]

for i, y in enumerate(item_ys):
    x0 = card_x
    y0 = y
    x1 = card_x + card_w
    y1 = y + card_h

    # subtle shadow (slightly larger rounded rect behind)
    shadow_offset = 4
    shadow_box = [x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset]
    try:
        draw.rounded_rectangle(shadow_box, radius=card_radius + 2, fill=card_shadow)
    except Exception:
        # fallback to normal rectangle if rounded not available
        draw.rectangle(shadow_box, fill=card_shadow)

    # main card background (white)
    try:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=card_radius, fill=card_bg)
    except Exception:
        draw.rectangle([x0, y0, x1, y1], fill=card_bg)

    # separator line at bottom of each card (inside the card)
    draw.line([(x0 + 24, y1 - 1), (x1 - 24, y1 - 1)], fill=separator, width=1)

# 4) Accent for selected item (first item "Anytime"): vertical accent stripe on the left of that card
selected_idx = 0
sel_y = item_ys[selected_idx]
accent_width = 12
draw.rectangle([(card_x + 8, sel_y + 12), (card_x + 8 + accent_width, sel_y + card_h - 12)], fill=accent_orange)

# 5) Large empty content area below the last card (keep white, but add very subtle horizontal separators)
content_top = item_ys[-1] + card_h + 24
for i in range(4):
    y_line = content_top + i * 220
    if y_line < H - 80:
        draw.line([(36, y_line), (W - 36, y_line)], fill=(250, 250, 252), width=1)

# 6) Bottom safe area divider
bottom_divider_y = H - 90
draw.line([(0, bottom_divider_y), (W, bottom_divider_y)], fill=toolbar_divider, width=1)

# (All actual icons and texts will be pasted on top at their detected bounding boxes; we only provided backgrounds/structure.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/00_icon_7.47.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["7.47"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/01_icon_7.47.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (115, 3), _c1)
except Exception:
    pass
layout["7.47"] = [115, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 64, 61)
    canvas.paste(_c2, (308, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/03_icon_7.47.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["7.47"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 63)
    canvas.paste(_c5, (1154, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 4, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 98, 61)
    canvas.paste(_c6, (1216, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1216, 3, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 58)
    canvas.paste(_c7, (248, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/09_icon_7.47.png
try:
    _c9 = get_crop(9, 90, 61)
    canvas.paste(_c9, (17, 4), _c9)
except Exception:
    pass
layout["7.47"] = [17, 4, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 123, 129)
    canvas.paste(_c10, (1291, 246), _c10)
except Exception:
    pass
layout["icon_10"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/11_icon_Tomorrow.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 594), _c11)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_08_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-10/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
