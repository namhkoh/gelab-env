# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_04
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-6.png
# step_index: 4/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page
# Available objects: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg = (250, 250, 252)            # overall page background (very light)
status_bar = (190, 190, 190)    # status bar gray
header_accent = (47, 84, 255)   # blue accent for underline/divider
muted_divider = (225, 227, 235) # subtle divider / card background
card_bg = (249, 250, 252)       # slightly off-white card backgrounds
panel_gray = (242, 244, 247)    # light panel area
shadow = (230, 232, 237)        # soft shadow / border color

# Fill overall background
draw.rectangle([0, 0, W, H], fill=bg)

# Status bar area (top)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_bar)

# Thin top divider under status bar (very subtle)
draw.rectangle([0, status_h - 1, W, status_h + 1], fill=shadow)

# Header area (toolbar) - keep it visually distinct but mostly same bg
header_top = status_h
header_bottom = 220
draw.rectangle([0, header_top, W, header_bottom], fill=bg)

# Blue underline divider for the "page title" area
underline_y = header_bottom - 20
underline_margin = 48
draw.rectangle([underline_margin, underline_y, W - underline_margin, underline_y + 6], fill=header_accent)

# Light horizontal divider below header
draw.rectangle([underline_margin, underline_y + 12, W - underline_margin, underline_y + 14], fill=muted_divider)

# Location/option card area (background behind "Nearby" / "Online events")
opt_card_top = underline_y + 36
opt_card_left = 40
opt_card_right = W - 40
opt_card_bottom = opt_card_top + 120
# rounded rectangle background
try:
    draw.rounded_rectangle([opt_card_left, opt_card_top, opt_card_right, opt_card_bottom],
                           radius=14, fill=card_bg, outline=shadow, width=1)
except Exception:
    # fallback if rounded_rectangle not available
    draw.rectangle([opt_card_left, opt_card_top, opt_card_right, opt_card_bottom], fill=card_bg, outline=shadow)

# Subtle inner separators for the two option columns (visual structure only)
col_x = W // 2
sep_top = opt_card_top + 18
sep_bottom = opt_card_bottom - 18
draw.line([col_x, sep_top, col_x, sep_bottom], fill=muted_divider, width=1)

# Placeholder small circular backgrounds for option icons (only as background shapes, very faint)
# Left circle position (do not draw icon content)
left_circle_center = (opt_card_left + 86, opt_card_top + (opt_card_bottom - opt_card_top)//2)
right_circle_center = (opt_card_left + (opt_card_right - opt_card_left)//2 + 86, left_circle_center[1])
circle_r = 36
draw.ellipse([left_circle_center[0]-circle_r, left_circle_center[1]-circle_r,
              left_circle_center[0]+circle_r, left_circle_center[1]+circle_r],
             fill=panel_gray, outline=None)
draw.ellipse([right_circle_center[0]-circle_r, right_circle_center[1]-circle_r,
              right_circle_center[0]+circle_r, right_circle_center[1]+circle_r],
             fill=panel_gray, outline=None)

# Big content area background (where event cards would appear) - large pale panel
content_top = opt_card_bottom + 40
content_left = 36
content_right = W - 36
content_bottom = H - 220
try:
    draw.rounded_rectangle([content_left, content_top, content_right, content_bottom],
                           radius=20, fill=(255,255,255), outline=muted_divider, width=1)
except Exception:
    draw.rectangle([content_left, content_top, content_right, content_bottom],
                   fill=(255,255,255), outline=muted_divider)

# Separator lines within content area to create structural rhythm (do not place text)
section_y = content_top + 220
for i in range(4):
    y = section_y + i * 260
    if y + 160 < content_bottom:
        # card background strip
        try:
            draw.rounded_rectangle([content_left + 16, y, content_right - 16, y + 180],
                                   radius=14, fill=card_bg, outline=shadow, width=1)
        except Exception:
            draw.rectangle([content_left + 16, y, content_right - 16, y + 180], fill=card_bg, outline=shadow)

# Bottom navigation area subtle bar (structure only)
bottom_nav_h = 110
draw.rectangle([0, H - bottom_nav_h, W, H], fill=bg)
draw.rectangle([0, H - bottom_nav_h - 2, W, H - bottom_nav_h], fill=muted_divider)

# Small central subtle dot indicating loading area background (not text)
# Place it near where "Loading" text will be overlaid by detected crop; this is a faint structural hint only
loading_center = (W//2, int(H*0.65))
draw.ellipse([loading_center[0]-6, loading_center[1]-6, loading_center[0]+6, loading_center[1]+6],
             fill=shadow)

# Final thin vertical margins lines for overall framing
draw.line([36, header_top, 36, H - bottom_nav_h], fill=muted_divider, width=1)
draw.line([W - 36, header_top, W - 36, H - bottom_nav_h], fill=muted_divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 93, 66)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1308, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 62, 62)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/03_icon_7.28.png
try:
    _c3 = get_crop(3, 61, 65)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["7.28"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/04_icon_7.28.png
try:
    _c4 = get_crop(4, 61, 65)
    canvas.paste(_c4, (114, 1), _c4)
except Exception:
    pass
layout["7.28"] = [114, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/05_icon_7.28.png
try:
    _c5 = get_crop(5, 168, 168)
    canvas.paste(_c5, (0, 72), _c5)
except Exception:
    pass
layout["7.28"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 56)
    canvas.paste(_c6, (250, 7), _c6)
except Exception:
    pass
layout["icon_6"] = [250, 7, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 81, 92)
    canvas.paste(_c7, (1313, 288), _c7)
except Exception:
    pass
layout["icon_7"] = [1313, 288, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 66)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/10_icon_7.28.png
try:
    _c10 = get_crop(10, 92, 64)
    canvas.paste(_c10, (15, 1), _c10)
except Exception:
    pass
layout["7.28"] = [15, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/11_text_New_York.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_04_2024_4_23_19_27_45f56b06f31541079045047b6d542613-6/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
