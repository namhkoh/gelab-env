# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_04
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6.png
# step_index: 4/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background base
bg = (255, 255, 255)
draw.rectangle((0, 0, 1440, 2960), fill=bg)

# Top status bar background (covers area behind status icons)
status_bar_h = 88
status_bar_color = (208, 208, 208)  # light gray similar to screenshot
draw.rectangle((0, 0, 1440, status_bar_h), fill=status_bar_color)

# Thin divider under status bar
draw.line([(0, status_bar_h), (1440, status_bar_h)], fill=(190, 190, 190), width=1)

# Header underline (blue accent under the page title area)
# Positioned to align visually under the header/title region in the screenshot
header_underline_y = 392
blue_accent = (51, 85, 255)
draw.line([(48, header_underline_y), (1392, header_underline_y)], fill=blue_accent, width=4)

# Subtle top-left back area background (keeps header area visually distinct)
header_bg_top = status_bar_h
header_bg_bottom = header_underline_y + 6
draw.rectangle((0, header_bg_top, 1440, header_bg_bottom), fill=(255, 255, 255))

# "Nearby" section card - rounded rectangle background behind the list item group
nearby_card_left = 48
nearby_card_top = 420
nearby_card_right = 1392
nearby_card_bottom = 540
nearby_card_radius = 14
nearby_card_fill = (249, 250, 255)  # very subtle off-white / bluish tint to separate from page
nearby_card_outline = (235, 236, 245)
draw.rounded_rectangle(
    (nearby_card_left, nearby_card_top, nearby_card_right, nearby_card_bottom),
    radius=nearby_card_radius,
    fill=nearby_card_fill,
    outline=nearby_card_outline,
    width=1
)

# Subtle shadow line below the nearby card to lift it off the page
draw.line(
    [(nearby_card_left + 8, nearby_card_bottom + 2), (nearby_card_right - 8, nearby_card_bottom + 2)],
    fill=(245, 246, 250),
    width=2
)

# Large content area background (main scrolling area)
content_top = nearby_card_bottom + 32
content_bottom = 2860
content_left = 0
content_right = 1440
# Keep it white but add a very faint cool tint band to match screenshot's neutral background
draw.rectangle((content_left, content_top, content_right, content_bottom), fill=(255, 255, 255))

# Subtle horizontal separators where content sections would be (do not draw icons/text)
sep_x1 = 48
sep_x2 = 1392
for y in (content_top + 280, content_top + 760, content_top + 1320):
    draw.line([(sep_x1, y), (sep_x2, y)], fill=(243, 244, 248), width=1)

# Small center guide (very faint) to indicate loading/content area - does not duplicate detected "Loading" text
# Draw a tiny faint dot (not text or icon) to reflect the subtle center mark in the screenshot
dot_x = 720
dot_y = 1440
draw.ellipse((dot_x - 2, dot_y - 2, dot_x + 2, dot_y + 2), fill=(240, 241, 246))

# Final thin page bottom divider
draw.line([(0, 2958), (1440, 2958)], fill=(230, 230, 235), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 69)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/01_icon_8.07.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["8.07"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/02_icon_8.07.png
try:
    _c2 = get_crop(2, 61, 64)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["8.07"] = [179, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 95, 65)
    canvas.paste(_c3, (1215, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1215, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/04_icon_8.07.png
try:
    _c4 = get_crop(4, 64, 67)
    canvas.paste(_c4, (111, 1), _c4)
except Exception:
    pass
layout["8.07"] = [111, 1, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 62)
    canvas.paste(_c5, (308, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 79, 86)
    canvas.paste(_c6, (1314, 291), _c6)
except Exception:
    pass
layout["icon_6"] = [1314, 291, 1393, 377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 56)
    canvas.paste(_c7, (249, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 6, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 58)
    canvas.paste(_c8, (1323, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1323, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/09_icon_8.07.png
try:
    _c9 = get_crop(9, 93, 60)
    canvas.paste(_c9, (14, 4), _c9)
except Exception:
    pass
layout["8.07"] = [14, 4, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/10_text_Chicago.png
try:
    _c10 = get_crop(10, 1344, 129)
    canvas.paste(_c10, (48, 264), _c10)
except Exception:
    pass
layout["Chicago"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/11_text_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/12_text_Current_location.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_04_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-6/13_text_Loading.png
try:
    _c13 = get_crop(13, 156, 55)
    canvas.paste(_c13, (641, 1970), _c13)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
