# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_02
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4.png
# step_index: 2/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for a 1440x2960 mobile canvas.
# Assumes available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Clear canvas to white (dominant color)
draw.rectangle([(0, 0), (w, h)], fill=(255, 255, 255))

# STATUS BAR (top area)
status_h = 80  # approximate status bar height
status_color = (200, 200, 200)  # light gray
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Subtle bottom stroke for status bar
draw.line([(0, status_h-1), (w, status_h-1)], fill=(190, 190, 190), width=1)

# HEADER / TOOLBAR background (below status bar)
header_top = status_h
header_h = 140
header_bottom = header_top + header_h
header_color = (250, 250, 252)  # slightly off-white
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_color)

# Subtle divider under header area (thin purple-ish line like screenshot)
divider_x0 = 48
divider_x1 = w - 48
divider_y = header_bottom - 14
divider_color = (195, 183, 205)  # muted lavender/purple
draw.line([(divider_x0, divider_y), (divider_x1, divider_y)], fill=divider_color, width=2)

# Rounded search card background behind the "Find events in..." area
search_card_top = header_top + 18
search_card_bottom = divider_y + 12
search_card_margin_lr = 40
search_card_rect = [
    (search_card_margin_lr, search_card_top),
    (w - search_card_margin_lr, search_card_bottom)
]
draw.rounded_rectangle(search_card_rect, radius=18, fill=(255, 255, 255), outline=(236, 232, 239), width=1)

# "Nearby" row card background (subtle separation)
nearby_top = search_card_bottom + 30
nearby_bottom = nearby_top + 120
nearby_margin_lr = 40
draw.rounded_rectangle(
    [(nearby_margin_lr, nearby_top), (w - nearby_margin_lr, nearby_bottom)],
    radius=14,
    fill=(255, 255, 255),
    outline=(245, 244, 247),
    width=1
)

# Thin separator line between "Nearby" and the next section
sep_y = nearby_bottom + 26
draw.line([(nearby_margin_lr + 8, sep_y), (w - nearby_margin_lr - 8, sep_y)], fill=(240, 239, 244), width=1)

# "Browsing in" / Selection area background (large clear area for selection)
browsing_top = sep_y + 18
browsing_bottom = browsing_top + 160
draw.rectangle([(0, browsing_top), (w, browsing_bottom)], fill=(255, 255, 255))

# Right-side circular selection background (area only, icons are auto-pasted so we only hint at background)
select_circle_center = (w - 120, browsing_top + 60)
select_circle_radius = 48
# draw a very faint circular background to match screenshot's soft check circle background
circle_bbox = [
    (select_circle_center[0] - select_circle_radius, select_circle_center[1] - select_circle_radius),
    (select_circle_center[0] + select_circle_radius, select_circle_center[1] + select_circle_radius)
]
draw.ellipse(circle_bbox, fill=(250, 248, 252))

# Large content area background (the rest of the screen remains white; add a faint vertical rhythm line)
content_top = browsing_bottom + 20
content_margin_lr = 48
# subtle vertical guideline blocks to create structure (very faint)
for y in range(content_top + 40, h - 200, 260):
    # simulate space for cards/posts without drawing actual content
    card_h = 220
    card_rect = [(content_margin_lr, y), (w - content_margin_lr, y + card_h)]
    draw.rounded_rectangle(card_rect, radius=12, fill=(255, 255, 255), outline=(245, 244, 247), width=1)

# Bottom-most faint divider to frame page end
draw.line([(0, h-1), (w, h-1)], fill=(230, 230, 230), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/00_icon_8.07.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["8.07"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/01_icon_8.07.png
try:
    _c1 = get_crop(1, 60, 63)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["8.07"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 62, 61)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/03_icon_8.07.png
try:
    _c3 = get_crop(3, 62, 65)
    canvas.paste(_c3, (112, 1), _c3)
except Exception:
    pass
layout["8.07"] = [112, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 56)
    canvas.paste(_c4, (249, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 6, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 103, 108)
    canvas.paste(_c5, (1291, 836), _c5)
except Exception:
    pass
layout["icon_5"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 47, 57)
    canvas.paste(_c6, (1322, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 74, 61)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1287, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 59)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/09_icon_8.07.png
try:
    _c9 = get_crop(9, 95, 60)
    canvas.paste(_c9, (12, 4), _c9)
except Exception:
    pass
layout["8.07"] = [12, 4, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/14_text_Browsing_in.png
try:
    _c14 = get_crop(14, 228, 55)
    canvas.paste(_c14, (44, 742), _c14)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/15_text_Online_events.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_02_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-4/16_text_Virtual_attendance.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
