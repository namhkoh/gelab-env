# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_07
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9.png
# step_index: 7/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page
# (Uses provided canvas and draw objects)

# Colors
bg_offwhite = (250, 252, 255)    # very slight blue-tinted white
status_bar_gray = (207, 207, 207)
divider_blue = (43, 96, 255)
muted_gray = (196, 196, 200)
card_bg = (255, 255, 255)
soft_section = (245, 250, 255)
separator_gray = (230, 230, 235)
shadow = (220, 220, 225)

w, h = canvas.size

# Base background
draw.rectangle([(0, 0), (w, h)], fill=bg_offwhite)

# Status bar area (top)
status_bar_h = 72
draw.rectangle([(0, 0), (w, status_bar_h)], fill=status_bar_gray)
# subtle bottom hairline for status bar
draw.line([(0, status_bar_h), (w, status_bar_h)], fill=muted_gray, width=1)

# Header / toolbar area beneath status bar
header_top = status_bar_h
header_bottom = 240
draw.rectangle([(0, header_top), (w, header_bottom)], fill=card_bg)

# Blue underline below the header (starts near typical title left margin)
underline_y = header_bottom - 20
left_margin = 48
right_margin = w - 48
draw.line([(left_margin, underline_y), (right_margin, underline_y)], fill=divider_blue, width=4)

# Thin divider under header for subtle separation
draw.line([(0, header_bottom), (w, header_bottom)], fill=separator_gray, width=1)

# Section row background for options ("Nearby", "Online events")
section_top = header_bottom + 40
section_bottom = section_top + 120
section_margin = 32
draw.rounded_rectangle(
    [(section_margin, section_top), (w - section_margin, section_bottom)],
    radius=12, fill=soft_section, outline=None
)

# Subtle separators around the options row
draw.line([(section_margin + 12, section_bottom + 8), (w - section_margin - 12, section_bottom + 8)], fill=separator_gray, width=1)

# Large content area background (events list area)
content_top = section_bottom + 32
content_margin = 32
draw.rectangle([(0, content_top), (w, h)], fill=bg_offwhite)

# Draw a few structural "card" backgrounds (rounded rectangles) for event list placeholders
card_w_left = content_margin
card_w_right = w - content_margin
card_height = 320
gap = 28
y = content_top + 20
for i in range(3):
    card_box = [(card_w_left, y), (card_w_right, y + card_height)]
    # card shadow
    shadow_box = [(card_w_left + 6, y + 6), (card_w_right + 6, y + card_height + 6)]
    draw.rounded_rectangle(shadow_box, radius=16, fill=shadow)
    # card background
    draw.rounded_rectangle(card_box, radius=16, fill=card_bg)
    # subtle divider within card near its bottom to indicate structured content area
    draw.line([(card_w_left + 24, y + card_height - 56), (card_w_right - 24, y + card_height - 56)], fill=separator_gray, width=1)
    y += card_height + gap

# Bottom area subtle divider
draw.line([(0, h - 160), (w, h - 160)], fill=separator_gray, width=1)

# Fine vertical guide accents (do not conflict with icons/text)
# these are faint and structural for layout feel
draw.line([(left_margin, header_top + 8), (left_margin, h - 8)], fill=(245,245,247), width=1)
draw.line([(right_margin, header_top + 8), (right_margin, h - 8)], fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 66)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1311, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 62)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/03_icon_7.10.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["7.10"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/04_icon_7.10.png
try:
    _c4 = get_crop(4, 61, 64)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["7.10"] = [179, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/05_icon_7.10.png
try:
    _c5 = get_crop(5, 59, 65)
    canvas.paste(_c5, (116, 1), _c5)
except Exception:
    pass
layout["7.10"] = [116, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 81, 92)
    canvas.paste(_c6, (1313, 288), _c6)
except Exception:
    pass
layout["icon_6"] = [1313, 288, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 57)
    canvas.paste(_c7, (249, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 65)
    canvas.paste(_c8, (1319, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1319, 0, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/09_icon_Nearby.png
try:
    _c9 = get_crop(9, 415, 114)
    canvas.paste(_c9, (48, 465), _c9)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 66)
    canvas.paste(_c10, (383, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/11_text_7.10.png
try:
    _c11 = get_crop(11, 89, 41)
    canvas.paste(_c11, (22, 17), _c11)
except Exception:
    pass
layout["7.10"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/12_text_Chicago.png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Chicago"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_07_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-9/15_text_Loading.png
try:
    _c15 = get_crop(15, 156, 55)
    canvas.paste(_c15, (641, 1970), _c15)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
