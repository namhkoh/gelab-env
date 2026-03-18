# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_08
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10.png
# step_index: 8/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background and structural elements for Eventbrite-like UI
# canvas: PIL Image (1440x2960), draw: ImageDraw

# Fill base background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")
# subtle bottom edge for status bar
draw.line([(0, status_h), (1440, status_h)], fill="#BFBFBF", width=1)

# Toolbar / header area (keeps white but add a subtle divide)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#F1F2F4", width=1)

# Blue underline (search input indicator) — will sit behind pasted text
underline_y = 320
draw.line([(48, underline_y), (1392, underline_y)], fill="#2B5BFF", width=4)

# Section cards area (rounded backgrounds behind "Nearby" and "Online events")
left_card = (28, 444, 488, 592)
right_card = (496, 444, 1016, 592)
card_fill = "#F5FBFF"   # very pale blue background
card_outline = None
corner_radius = 24

# Draw rounded rectangles for the two option cards
draw.rounded_rectangle(left_card, radius=corner_radius, fill=card_fill, outline=card_outline)
draw.rounded_rectangle(right_card, radius=corner_radius, fill=card_fill, outline=card_outline)

# Subtle inner separators / shadows under cards
draw.line([(left_card[0]+8, left_card[3]), (left_card[2]-8, left_card[3])], fill="#E6F0FF", width=2)
draw.line([(right_card[0]+8, right_card[3]), (right_card[2]-8, right_card[3])], fill="#E6F0FF", width=2)

# Horizontal section divider below the option cards
divider_y = 624
draw.line([(24, divider_y), (1416, divider_y)], fill="#F2F3F6", width=1)

# Large content area background (keeps white but add a very faint cool tint band near center to suggest content region)
band_top = 900
band_bottom = 1300
draw.rectangle([(0, band_top), (1440, band_bottom)], fill="#FFFFFF")  # left white; keep subtle for pasted content

# Optional light footer divider near bottom to indicate end of content zone
footer_div_y = 2700
draw.line([(24, footer_div_y), (1416, footer_div_y)], fill="#F1F2F4", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 96, 66)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/02_icon_9.12.png
try:
    _c2 = get_crop(2, 58, 64)
    canvas.paste(_c2, (180, 1), _c2)
except Exception:
    pass
layout["9.12"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/03_icon_9.12.png
try:
    _c3 = get_crop(3, 52, 65)
    canvas.paste(_c3, (117, 1), _c3)
except Exception:
    pass
layout["9.12"] = [117, 1, 169, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 80, 90)
    canvas.paste(_c4, (1314, 289), _c4)
except Exception:
    pass
layout["icon_4"] = [1314, 289, 1394, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/05_icon_9.12.png
try:
    _c5 = get_crop(5, 168, 168)
    canvas.paste(_c5, (0, 72), _c5)
except Exception:
    pass
layout["9.12"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 64)
    canvas.paste(_c6, (1320, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1320, 0, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 63)
    canvas.paste(_c7, (315, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 54, 62)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/09_icon_Nearby.png
try:
    _c9 = get_crop(9, 415, 114)
    canvas.paste(_c9, (48, 465), _c9)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 46, 65)
    canvas.paste(_c10, (384, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [384, 0, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/11_text_9.12.png
try:
    _c11 = get_crop(11, 91, 43)
    canvas.paste(_c11, (20, 17), _c11)
except Exception:
    pass
layout["9.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/12_text_Los_Angeles.png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_08_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-10/15_text_Loading.png
try:
    _c15 = get_crop(15, 156, 55)
    canvas.paste(_c15, (641, 1970), _c15)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
