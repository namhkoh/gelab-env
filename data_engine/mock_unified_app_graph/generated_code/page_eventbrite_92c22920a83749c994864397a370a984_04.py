# page_id: page_eventbrite_92c22920a83749c994864397a370a984_04
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-6.png
# step_index: 4/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page.
# Assumes: canvas (1440x2960 RGB PIL Image) and draw (ImageDraw) are provided.

# Colors
status_bar_color = (200, 200, 200)     # light grey for status bar
accent_blue = (50, 96, 255)            # bright app blue for underline/highlights
light_accent = (234, 244, 255)         # very light blue for chips/badges
divider_gray = (220, 220, 225)         # subtle divider gray
card_bg = (249, 249, 251)              # off-white card background
placeholder_gray = (240, 242, 245)     # event placeholder background

W, H = canvas.size

# Status bar area (top)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Thin bottom divider under status bar
draw.line([(0, status_h), (W, status_h)], fill=divider_gray, width=1)

# Header area (keeps canvas white; we draw a thin blue underline as the header divider)
# Blue underline below the header/title
underline_y = 360
underline_left = 48
underline_right = W - 48
underline_thickness = 4
draw.rectangle([(underline_left, underline_y), (underline_right, underline_y + underline_thickness)],
               fill=accent_blue)

# Subtle hairline above underline for slight separation
draw.line([(underline_left, underline_y - 6), (underline_right, underline_y - 6)],
          fill=(235,235,240), width=1)

# Section chips / option backgrounds (rounded rectangles behind groups of elements)
# Left chip (e.g., "Nearby" group background)
left_chip = (48, 465, 48 + 415, 465 + 114)
draw.rounded_rectangle(left_chip, radius=28, fill=light_accent)

# Right chip (e.g., "Online events" group background)
right_chip = (511, 465, 511 + 452, 465 + 114)
draw.rounded_rectangle(right_chip, radius=28, fill=light_accent)

# Subtle separators: a faint divider line below the chips to separate header from content
sep_y = 465 + 114 + 24
draw.line([(48, sep_y), (W - 48, sep_y)], fill=divider_gray, width=1)

# Content area placeholders (representing event cards / image backgrounds)
# First large content card placeholder
card1 = (48, 760, W - 48, 1160)
draw.rounded_rectangle(card1, radius=20, fill=placeholder_gray)

# Thin inner divider on card1 to indicate image/content split (decorative, not duplicating icons/text)
draw.line([(48 + 24, 940), (W - 48 - 24, 940)], fill=(245,246,248), width=1)

# Second content card placeholder
card2 = (48, 1260, W - 48, 1660)
draw.rounded_rectangle(card2, radius=20, fill=placeholder_gray)

# Soft divider between cards
draw.line([(48, 1160 + 40), (W - 48, 1160 + 40)], fill=(245,245,247), width=1)

# Footer subtle background band near bottom (keeps the page feeling grounded)
footer_top = H - 160
draw.rectangle([(0, footer_top), (W, H)], fill=(255, 255, 255))

# Very subtle center spot where a loading indicator will appear (only a faint backdrop, not the spinner/text)
loading_center = (W // 2, 1970 + 20)
dot_radius = 6
draw.ellipse([(loading_center[0] - dot_radius, loading_center[1] - dot_radius),
              (loading_center[0] + dot_radius, loading_center[1] + dot_radius)],
             fill=(242, 243, 246))

# Final subtle vignette lines to frame the content area (very light)
draw.line([(48, 720), (W - 48, 720)], fill=(250,250,251), width=1)
draw.line([(48, 1680), (W - 48, 1680)], fill=(250,250,251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/01_icon_4.59.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["4.59"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 62)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 92, 66)
    canvas.paste(_c3, (1214, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1214, 0, 1306, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/04_icon_4.59.png
try:
    _c4 = get_crop(4, 61, 65)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["4.59"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/05_icon_4.59.png
try:
    _c5 = get_crop(5, 61, 66)
    canvas.paste(_c5, (114, 1), _c5)
except Exception:
    pass
layout["4.59"] = [114, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 57)
    canvas.paste(_c6, (250, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [250, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 81, 92)
    canvas.paste(_c7, (1313, 288), _c7)
except Exception:
    pass
layout["icon_7"] = [1313, 288, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 66)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/10_icon_4.59.png
try:
    _c10 = get_crop(10, 92, 64)
    canvas.paste(_c10, (15, 1), _c10)
except Exception:
    pass
layout["4.59"] = [15, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/11_text_Chicago.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Chicago"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_04_2024_4_24_16_59_92c22920a83749c994864397a370a984-6/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
