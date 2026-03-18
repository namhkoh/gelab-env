# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_08
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10.png
# step_index: 8/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page
# (Uses provided variables: canvas (1440x2960), draw (ImageDraw), font_*)

# Canvas background (dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~72px) - subtle muted gray to match screenshot status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(242, 242, 242))

# Header area (toolbar) below status bar
header_y0 = status_h
header_y1 = 160
draw.rectangle([(0, header_y0), (1440, header_y1)], fill=(255, 255, 255))

# Header underline (prominent blue rule under the page title)
underline_y = header_y1 - 2
draw.line([(48, underline_y), (1392, underline_y)], fill=(45, 89, 255), width=4)

# Very subtle hairline divider under the blue rule
draw.line([(48, underline_y + 6), (1392, underline_y + 6)], fill=(232, 232, 236), width=1)

# Section row group background (e.g., the "Nearby" row container)
row_x0 = 32
row_x1 = 1408
row_y0 = 220
row_y1 = 332
corner_radius = 12
# light off-white/very pale blue card fill to separate from canvas
try:
    draw.rounded_rectangle([(row_x0, row_y0), (row_x1, row_y1)],
                           radius=corner_radius,
                           fill=(250, 252, 255),
                           outline=(235, 236, 241),
                           width=1)
except AttributeError:
    # Fallback if rounded_rectangle not available: draw rectangle with slight inset outlines
    draw.rectangle([(row_x0, row_y0), (row_x1, row_y1)], fill=(250, 252, 255), outline=(235, 236, 241))

# Thin separator line below the section row
sep_y = row_y1 + 12
draw.line([(48, sep_y), (1392, sep_y)], fill=(238, 238, 242), width=1)

# Additional faint section divider further down for content grouping
second_sep_y = 420
draw.line([(48, second_sep_y), (1392, second_sep_y)], fill=(246, 246, 248), width=1)

# Large content area card (empty / placeholder for posts or images) with subtle tint
content_x0 = 48
content_x1 = 1392
content_y0 = 480
content_y1 = 1100
try:
    draw.rounded_rectangle([(content_x0, content_y0), (content_x1, content_y1)],
                           radius=16,
                           fill=(255, 255, 255),
                           outline=(240, 241, 245),
                           width=1)
except AttributeError:
    draw.rectangle([(content_x0, content_y0), (content_x1, content_y1)], fill=(255, 255, 255),
                   outline=(240, 241, 245))

# Subtle shadow line under the content card to give depth
draw.line([(content_x0 + 4, content_y1 + 2), (content_x1 - 4, content_y1 + 2)], fill=(247, 247, 249), width=2)

# Bottom region separators to indicate page end/loading zone
bottom_sep_y = 1900
draw.line([(48, bottom_sep_y), (1392, bottom_sep_y)], fill=(245, 245, 247), width=1)

# Final subtle horizontal guide near center to help placement of pasted spinner/text
center_guide_y = 1920
draw.line([(360, center_guide_y), (1080, center_guide_y)], fill=(255, 255, 255), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 69)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 95, 65)
    canvas.paste(_c1, (1216, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1216, 0, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/02_icon_9.20.png
try:
    _c2 = get_crop(2, 59, 64)
    canvas.paste(_c2, (178, 1), _c2)
except Exception:
    pass
layout["9.20"] = [178, 1, 237, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 79, 85)
    canvas.paste(_c3, (1314, 291), _c3)
except Exception:
    pass
layout["icon_3"] = [1314, 291, 1393, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/04_icon_9.20.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["9.20"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/05_icon_9.20.png
try:
    _c5 = get_crop(5, 53, 64)
    canvas.paste(_c5, (116, 1), _c5)
except Exception:
    pass
layout["9.20"] = [116, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 56, 63)
    canvas.paste(_c6, (246, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [246, 1, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 58)
    canvas.paste(_c7, (1323, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1323, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 64)
    canvas.paste(_c8, (315, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [315, 1, 368, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/09_text_9.20.png
try:
    _c9 = get_crop(9, 91, 43)
    canvas.paste(_c9, (20, 17), _c9)
except Exception:
    pass
layout["9.20"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/10_text_New_York.png
try:
    _c10 = get_crop(10, 1344, 129)
    canvas.paste(_c10, (48, 264), _c10)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/11_text_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/12_text_Current_location.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_08_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-10/13_text_Loading.png
try:
    _c13 = get_crop(13, 156, 55)
    canvas.paste(_c13, (641, 1970), _c13)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
