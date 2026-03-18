# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_05
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7.png
# step_index: 5/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#F3F5F7")

# Subtle divider under status bar
draw.line([(24, status_h), (1440-24, status_h)], fill="#E6E9ED", width=1)

# Header underline (accent) below the page title area
# Positioned below the detected title bounding box to avoid overlapping pasted text.
underline_y = 384
draw.line([(48, underline_y), (1440-48, underline_y)], fill="#2D57FF", width=6)

# Thin soft highlight just below the accent line
draw.line([(48, underline_y + 8), (1440-48, underline_y + 8)], fill="#EEF4FF", width=1)

# Rounded card area behind the two option chips ("Nearby" / "Online events")
chips_card_top = 420
chips_card_bottom = 560
draw.rounded_rectangle(
    [(32, chips_card_top), (1440-32, chips_card_bottom)],
    radius=18,
    fill="#FFFFFF",
    outline="#EEF3FF",
    width=1
)

# Large content region/card area (main feed/background for events)
content_top = chips_card_bottom + 40
content_bottom = 1960
draw.rectangle(
    [(24, content_top), (1440-24, content_bottom)],
    fill="#FFFFFF"
)

# Subtle separator above the loading area
draw.line([(48, content_bottom), (1440-48, content_bottom)], fill="#F1F3F6", width=1)

# Very faint full-bleed tint at top area to ground the header visually
draw.rectangle([(0, status_h), (1440, chips_card_top)], fill=None, outline=None)
# Add a very subtle gradient-like band using several horizontal lines for depth
for i, alpha_offset in enumerate([0, 1, 2, 3, 4]):
    y = status_h + 6 + i*10
    color = (236 + alpha_offset*2, 240 + alpha_offset*2, 255)  # very light bluish tint
    draw.line([(0, y), (1440, y)], fill=color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 69)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 94, 66)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1309, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/02_icon_9.41.png
try:
    _c2 = get_crop(2, 59, 64)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["9.41"] = [179, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/03_icon_9.41.png
try:
    _c3 = get_crop(3, 55, 65)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["9.41"] = [114, 1, 169, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/04_icon_9.41.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["9.41"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 80, 90)
    canvas.paste(_c5, (1314, 289), _c5)
except Exception:
    pass
layout["icon_5"] = [1314, 289, 1394, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 53, 64)
    canvas.paste(_c6, (315, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [315, 1, 368, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 55, 63)
    canvas.paste(_c7, (246, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [246, 1, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 47, 65)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/10_text_9.41.png
try:
    _c10 = get_crop(10, 93, 50)
    canvas.paste(_c10, (18, 12), _c10)
except Exception:
    pass
layout["9.41"] = [18, 12, 111, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/11_text_Los_Angeles.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_05_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-7/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
