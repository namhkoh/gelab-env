# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_04
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6.png
# step_index: 4/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill background
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top)
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill=(214, 214, 214))
# subtle bottom divider of status bar
draw.line((0, status_bar_h, 1440, status_bar_h), fill=(200, 200, 200), width=1)

# Header area (title area background is same white, but draw structural divider)
header_top = status_bar_h + 40
header_bottom = header_top + 220
# blue underline/divider under header (matches app accent)
blue_accent = (42, 82, 255)
underline_y = header_bottom - 12
draw.line((48, underline_y, 1440 - 48, underline_y), fill=blue_accent, width=4)

# thin subtle top divider for header region
draw.line((48, header_top - 18, 1440 - 48, header_top - 18), fill=(242, 243, 248), width=1)

# Nearby section card background (subtle rounded rect behind the list item)
card_x0 = 40
card_x1 = 1400
card_y0 = underline_y + 36
card_y1 = card_y0 + 120
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=14, fill=(249, 250, 255), outline=None)

# Small separator line below the card
sep_y = card_y1 + 18
draw.line((48, sep_y, 1440 - 48, sep_y), fill=(238, 239, 243), width=1)

# Large content area background (keeps white but add a very subtle tint block to indicate content region)
content_top = sep_y + 20
content_bottom = 2200
draw.rectangle((48, content_top, 1440 - 48, content_bottom), fill=(255, 255, 255))

# Subtle circular background for the loading area (structural, not the spinner)
loading_center_x = 720
loading_center_y = 1700
loading_radius = 240
draw.ellipse((loading_center_x - loading_radius, loading_center_y - loading_radius,
              loading_center_x + loading_radius, loading_center_y + loading_radius),
             outline=(235, 235, 240), width=6)

# Additional faint horizontal separators for future sections
for offset in (content_top + 260, content_top + 520, content_top + 900):
    draw.line((48, offset, 1440 - 48, offset), fill=(245, 246, 249), width=1)

# Bottom safe-area line
draw.line((0, 2950, 1440, 2950), fill=(240, 240, 245), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 70)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/01_icon_5.34.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["5.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 97, 66)
    canvas.paste(_c2, (1214, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1214, 0, 1311, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/03_icon_5.34.png
try:
    _c3 = get_crop(3, 60, 64)
    canvas.paste(_c3, (180, 1), _c3)
except Exception:
    pass
layout["5.34"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 62, 62)
    canvas.paste(_c4, (309, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [309, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 80, 88)
    canvas.paste(_c5, (1313, 290), _c5)
except Exception:
    pass
layout["icon_5"] = [1313, 290, 1393, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/06_icon_5.34.png
try:
    _c6 = get_crop(6, 61, 66)
    canvas.paste(_c6, (113, 1), _c6)
except Exception:
    pass
layout["5.34"] = [113, 1, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 56)
    canvas.paste(_c7, (250, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [250, 6, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/09_icon_Loading.png
try:
    _c9 = get_crop(9, 355, 430)
    canvas.paste(_c9, (543, 1610), _c9)
except Exception:
    pass
layout["Loading"] = [543, 1610, 898, 2040]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/10_icon_5.34.png
try:
    _c10 = get_crop(10, 93, 64)
    canvas.paste(_c10, (15, 1), _c10)
except Exception:
    pass
layout["5.34"] = [15, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 50, 66)
    canvas.paste(_c11, (383, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/12_text_Los_Angeles.png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/13_text_Nearby.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_04_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]
