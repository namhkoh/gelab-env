# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_05
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7.png
# step_index: 5/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 84)], fill=(132, 132, 132))

# subtle bottom shadow for status bar
for i, alpha in enumerate([20, 15, 10], start=0):
    y = 84 + i
    draw.line([(0, y), (1440, y)], fill=(200 - i*6, 198 - i*6, 203 - i*6))

# Header / toolbar background area (beneath status bar)
draw.rectangle([(0, 84), (1440, 240)], fill=(255, 255, 255))

# Thin divider line under the main heading area (matches the thin underline in screenshot)
divider_y = 384
draw.rectangle([(48, divider_y), (1440-48, divider_y+2)], fill=(200, 196, 205))

# Card/background behind the "Nearby" row (rounded rectangle)
nearby_card_bbox = (32, 420, 1408, 560)
draw.rounded_rectangle(nearby_card_bbox, radius=18, fill=(249, 252, 255), outline=(230, 233, 240), width=1)

# Subtle separator under the Nearby card
sep_y = nearby_card_bbox[3] + 18
draw.line([(32, sep_y), (1408, sep_y)], fill=(240, 238, 242), width=1)

# "Browsing in" section background (a soft grouping area)
browsing_card_bbox = (32, 720, 1408, 940)
draw.rounded_rectangle(browsing_card_bbox, radius=22, fill=(250, 247, 255), outline=(236, 230, 243), width=1)

# Light horizontal separator between browsing header and content area
draw.line([(32, 816), (1408, 816)], fill=(245, 242, 247), width=1)

# Large content area background (main white content area - subtle off-white to match screenshot tone)
content_area_bbox = (0, 940, 1440, 2960)
draw.rectangle(content_area_bbox, fill=(255, 255, 255))

# Additional faint divider near top-left under the heading text area
draw.line([(48, 360), (1392, 360)], fill=(229, 226, 232), width=1)

# gentle vignette top-left to match subtle UI shading (very faint)
for i in range(6):
    alpha = 4 - i // 2
    y = 240 + i
    draw.line([(0, y), (1440, y)], fill=(250 - i, 249 - i, 251 - i))

# Right-side visual balance stripe (very faint, not an icon)
draw.rectangle([(1360, 240), (1440, 2960)], fill=(255, 255, 255, 10))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/00_icon_7.05.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["7.05"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/01_icon_7.05.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["7.05"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 62, 61)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/03_icon_7.05.png
try:
    _c3 = get_crop(3, 62, 66)
    canvas.paste(_c3, (113, 1), _c3)
except Exception:
    pass
layout["7.05"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 57)
    canvas.paste(_c4, (248, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [248, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (1321, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1321, 4, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 103, 108)
    canvas.paste(_c6, (1291, 836), _c6)
except Exception:
    pass
layout["icon_6"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 77, 61)
    canvas.paste(_c7, (1212, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 1, 1289, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 59)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 65)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/10_text_Find_events_in..png
try:
    _c10 = get_crop(10, 1344, 129)
    canvas.paste(_c10, (48, 264), _c10)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/11_text_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/12_text_Current_location.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/13_text_Browsing_in.png
try:
    _c13 = get_crop(13, 228, 55)
    canvas.paste(_c13, (44, 742), _c13)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/14_text_Online_events.png
try:
    _c14 = get_crop(14, 1440, 138)
    canvas.paste(_c14, (0, 816), _c14)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_05_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-7/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
