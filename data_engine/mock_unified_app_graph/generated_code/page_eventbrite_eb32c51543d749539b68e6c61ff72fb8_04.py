# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_04
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6.png
# step_index: 4/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & structural layout for the Eventbrite-like UI
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background with a very light off-white (matches screenshot's white/very-light tint)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 251, 253))

# Status bar area (top ~72px) - slightly darker to represent system/status bar
STATUS_BAR_H = 72
draw.rectangle((0, 0, 1440, STATUS_BAR_H), fill=(190, 190, 190))

# Thin subtle divider under status bar
draw.line((0, STATUS_BAR_H, 1440, STATUS_BAR_H), fill=(210, 210, 210), width=1)

# Toolbar / header background (just below status bar)
TOOLBAR_TOP = STATUS_BAR_H
TOOLBAR_BOTTOM = 168
draw.rectangle((0, TOOLBAR_TOP, 1440, TOOLBAR_BOTTOM), fill=(255, 255, 255))

# Subtle shadow line under toolbar
draw.line((0, TOOLBAR_BOTTOM, 1440, TOOLBAR_BOTTOM), fill=(235, 235, 235), width=2)

# Accent divider (blue) representing the header underline area (placed below toolbar,
# but avoid drawing inside detected "San Francisco" text bounding box by keeping it above that area)
ACCENT_Y = 180
draw.line((48, ACCENT_Y, 1392, ACCENT_Y), fill=(46, 99, 255), width=4)

# Section separator line below the small header/controls area
SEPARATOR_Y = 620
draw.line((24, SEPARATOR_Y, 1416, SEPARATOR_Y), fill=(240, 240, 244), width=1)

# Large content card background (rounded rectangle) for the results area
CARD_LEFT = 48
CARD_TOP = 660
CARD_RIGHT = 1392
CARD_BOTTOM = 1560
CARD_RADIUS = 28
draw.rounded_rectangle(
    (CARD_LEFT, CARD_TOP, CARD_RIGHT, CARD_BOTTOM),
    radius=CARD_RADIUS,
    fill=(249, 250, 252),
    outline=(235, 235, 240),
    width=1
)

# Add a faint inner horizontal rule to suggest subdivisions inside the content card
inner_line_y = CARD_TOP + 120
draw.line((CARD_LEFT + 24, inner_line_y, CARD_RIGHT - 24, inner_line_y), fill=(245, 245, 248), width=1)

# Secondary large dark content area to represent image/listing region lower on the page
# (placed below the rounded card and above the loading spinner area)
CONTENT_BLOCK_TOP = CARD_BOTTOM + 40
CONTENT_BLOCK_BOTTOM = 1560 + 300
draw.rectangle((48, CONTENT_BLOCK_TOP, 1392, CONTENT_BLOCK_BOTTOM), fill=(247, 248, 250))

# Subtle drop shadow under content block
draw.line((48, CONTENT_BLOCK_BOTTOM, 1392, CONTENT_BLOCK_BOTTOM), fill=(237, 237, 242), width=2)

# Bottom area hint (footer/background extension)
draw.rectangle((0, CONTENT_BLOCK_BOTTOM + 40, 1440, 2960), fill=(250, 251, 253))

# Edge gutters/guide lines (very faint) to frame content columns
draw.line((48, 0, 48, 2960), fill=(250, 250, 252), width=1)
draw.line((1392, 0, 1392, 2960), fill=(250, 250, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/01_icon_Loading.png
try:
    _c1 = get_crop(1, 366, 407)
    canvas.paste(_c1, (542, 1637), _c1)
except Exception:
    pass
layout["Loading"] = [542, 1637, 908, 2044]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 97, 66)
    canvas.paste(_c2, (1214, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1214, 0, 1311, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 62, 63)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 371, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/04_icon_7.47.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.47"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/05_icon_7.47.png
try:
    _c5 = get_crop(5, 61, 66)
    canvas.paste(_c5, (179, 1), _c5)
except Exception:
    pass
layout["7.47"] = [179, 1, 240, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/06_icon_7.47.png
try:
    _c6 = get_crop(6, 61, 66)
    canvas.paste(_c6, (113, 1), _c6)
except Exception:
    pass
layout["7.47"] = [113, 1, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 80, 92)
    canvas.paste(_c7, (1314, 288), _c7)
except Exception:
    pass
layout["icon_7"] = [1314, 288, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 58)
    canvas.paste(_c8, (249, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [249, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (1319, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 0, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 67)
    canvas.paste(_c10, (383, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 1, 433, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/11_icon_7.47.png
try:
    _c11 = get_crop(11, 91, 64)
    canvas.paste(_c11, (16, 1), _c11)
except Exception:
    pass
layout["7.47"] = [16, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/12_icon_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/13_text_San_Francisco.png
try:
    _c13 = get_crop(13, 1344, 129)
    canvas.paste(_c13, (48, 264), _c13)
except Exception:
    pass
layout["San_Francisco"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/14_text_Online_events.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_04_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]
