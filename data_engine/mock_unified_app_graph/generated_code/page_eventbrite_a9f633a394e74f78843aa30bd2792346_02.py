# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_02
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4.png
# step_index: 2/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#D3D3D3")  # light gray status bar

# Subtle hairline under status bar to separate from header
draw.line([(0, status_h), (1440, status_h)], fill="#CFC7D6", width=1)

# Header back area (keeps background, don't draw arrow or text)
header_top = status_h
header_h = 120
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill="#FFFFFF")

# Divider below the main header/title area (thin purple/gray line)
divider_y = header_top + header_h + 72
draw.line([(48, divider_y), (1392, divider_y)], fill="#D7CFDB", width=2)

# Section background for the "Nearby / Online events" row (rounded card)
sec_left = 32
sec_top = header_top + 140
sec_right = 1408
sec_bottom = sec_top + 200
draw.rounded_rectangle(
    [(sec_left, sec_top), (sec_right, sec_bottom)],
    radius=18,
    fill="#FAFBFF",
    outline="#EFEFF4",
    width=1
)

# Subtle separator between the card and the following content
sep_y = sec_bottom + 36
draw.line([(40, sep_y), (1400, sep_y)], fill="#F0EDF3", width=1)

# Light background block behind the browsing/location area to give structure
browse_top = sep_y + 28
browse_bottom = browse_top + 220
draw.rounded_rectangle(
    [(40, browse_top), (1400, browse_bottom)],
    radius=14,
    fill="#FFFFFF",
    outline="#F4F2F6",
    width=1
)

# A faint vertical guide on the left to suggest content column alignment
draw.line([(48, browse_top + 12), (48, 2800)], fill="#F2EFF4", width=2)

# Bottom area left intentionally blank (main content) with very subtle tint
content_top = browse_bottom + 20
draw.rectangle([(0, content_top), (1440, 2960)], fill="#FFFFFF")

# Small decorative underline under the "Find events in..." title area (accent)
accent_y = header_top + 120
draw.line([(48, accent_y), (360, accent_y)], fill="#AFA3C0", width=4)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (310, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/02_icon_4.50.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["4.50"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/03_icon_4.50.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["4.50"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/04_icon_4.50.png
try:
    _c4 = get_crop(4, 60, 66)
    canvas.paste(_c4, (115, 1), _c4)
except Exception:
    pass
layout["4.50"] = [115, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 57)
    canvas.paste(_c5, (250, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [250, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 45, 59)
    canvas.paste(_c6, (1323, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1323, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 63)
    canvas.paste(_c8, (1271, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1271, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 66)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/10_icon_Nearby.png
try:
    _c10 = get_crop(10, 415, 114)
    canvas.paste(_c10, (48, 465), _c10)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/11_text_4.50.png
try:
    _c11 = get_crop(11, 89, 43)
    canvas.paste(_c11, (22, 17), _c11)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_02_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-4/16_text_Washington.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Washington"] = [0, 816, 1440, 954]
