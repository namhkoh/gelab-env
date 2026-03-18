# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_05
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7.png
# step_index: 5/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw backgrounds and structure for the mobile UI (canvas and draw are provided)

# Base background (slightly off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 252))

# Status bar area (top ~72px) - a subtle grey strip
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(180, 180, 180))
# thin darker divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(160, 160, 160), width=1)

# Header/Search area background block (behind detected search elements)
header_top = status_h
header_bottom = 264  # approximate bottom of header area from screenshot
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(250, 250, 252))
# subtle divider under header
draw.line([(48, header_bottom), (1392, header_bottom)], fill=(225, 225, 228), width=2)

# Location row / small section gap (leave content to be pasted on top)
location_row_top = header_bottom + 8
location_row_bottom = 360
draw.rectangle([(0, location_row_top), (1440, location_row_bottom)], fill=(250, 250, 252))

# Separator under filter chips area
filter_sep_y = 540
draw.line([(48, filter_sep_y), (1392, filter_sep_y)], fill=(235, 235, 238), width=1)

# Large event card 1 background (rounded rectangle)
card1_x1, card1_y1 = 48, 620
card1_x2, card1_y2 = 48 + 1344, 1760  # extends down to make room for image + text
card_radius = 28
# subtle card shadow (simulated with a light rounded rect behind)
shadow_offset = 10
draw.rounded_rectangle(
    [(card1_x1 + shadow_offset, card1_y1 + shadow_offset),
     (card1_x2 + shadow_offset, card1_y2 + shadow_offset)],
    radius=card_radius, fill=(240, 240, 244)
)
# card surface
draw.rounded_rectangle([(card1_x1, card1_y1), (card1_x2, card1_y2)], radius=card_radius, fill=(255, 255, 255))
# thin border to separate from background
draw.rounded_rectangle([(card1_x1, card1_y1), (card1_x2, card1_y2)], radius=card_radius, outline=(235, 235, 238), width=1)

# Small separator between card1 and next content
draw.line([(48, card1_y2 + 18), (1392, card1_y2 + 18)], fill=(235, 235, 238), width=1)

# Large event card 2 background (rounded rectangle)
card2_x1, card2_y1 = 48, 1888
card2_x2, card2_y2 = 48 + 1344, 2708
draw.rounded_rectangle(
    [(card2_x1 + shadow_offset, card2_y1 + shadow_offset),
     (card2_x2 + shadow_offset, card2_y2 + shadow_offset)],
    radius=card_radius, fill=(240, 240, 244)
)
draw.rounded_rectangle([(card2_x1, card2_y1), (card2_x2, card2_y2)], radius=card_radius, fill=(255, 255, 255))
draw.rounded_rectangle([(card2_x1, card2_y1), (card2_x2, card2_y2)], radius=card_radius, outline=(235, 235, 238), width=1)

# Divider lines for list and content separation
# Under the "4,140 events" heading area (approx)
draw.line([(48, 560), (1392, 560)], fill=(230, 230, 233), width=1)

# A faint horizontal rule above the bottom navigation
bottom_nav_top = 2804
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill=(220, 220, 223), width=2)
# Bottom navigation background (leave icons area empty; only draw background)
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill=(255, 255, 255))
# subtle top shadow for nav
draw.line([(0, bottom_nav_top + 1), (1440, bottom_nav_top + 1)], fill=(230, 230, 233), width=1)

# Accent background band behind floating action areas on the right (to match screenshot's purple floating button area placement)
fab_band_x1, fab_band_x2 = 1080, 1440
fab_band_y1, fab_band_y2 = 2400, 2760
draw.rectangle([(fab_band_x1, fab_band_y1), (fab_band_x2, fab_band_y2)], fill=(255, 255, 255))  # keep white but ensure clear area

# Final subtle overall tint near top to echo app's soft neutral tone (very light)
draw.rectangle([(0, status_h), (1440, status_h + 6)], fill=(245, 245, 247))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/01_icon_Arts.png
try:
    _c1 = get_crop(1, 152, 135)
    canvas.paste(_c1, (850, 390), _c1)
except Exception:
    pass
layout["Arts"] = [850, 390, 1002, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/03_icon_An_East.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["An_East"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/04_icon_OAKERSON.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2434), _c4)
except Exception:
    pass
layout["OAKERSON"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/05_icon_OAKERSON.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2434), _c5)
except Exception:
    pass
layout["OAKERSON"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/07_icon_9.25.png
try:
    _c7 = get_crop(7, 123, 114)
    canvas.paste(_c7, (55, 114), _c7)
except Exception:
    pass
layout["9.25"] = [55, 114, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/09_icon_3388.png
try:
    _c9 = get_crop(9, 1344, 1194)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["3388"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/10_icon_9.25.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.25"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/11_icon_The_Good_Mood_Comedy_Show.png
try:
    _c11 = get_crop(11, 1344, 1194)
    canvas.paste(_c11, (48, 676), _c11)
except Exception:
    pass
layout["The_Good_Mood_Comedy_Show"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 68, 60)
    canvas.paste(_c12, (1209, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1209, 0, 1277, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 62, 59)
    canvas.paste(_c13, (1318, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1318, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 59, 62)
    canvas.paste(_c14, (312, 1), _c14)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/15_icon_9.25.png
try:
    _c15 = get_crop(15, 56, 63)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["9.25"] = [114, 0, 170, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/16_icon_Pop_Up_-_Stand_Up_Comedy_Series.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["Pop_Up_-_Stand_Up_:_Comed"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/18_icon_The_Good_Mood_Comedy_Show.png
try:
    _c18 = get_crop(18, 1344, 1194)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["The_Good_Mood_Comedy_Show"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/19_icon_Pop_Up_-_Stand_Up_Comedy_Series.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["Pop_Up_-_Stand_Up_:_Comed"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/20_icon_New_York.png
try:
    _c20 = get_crop(20, 434, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 49, 62)
    canvas.paste(_c21, (383, 1), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 40, 58)
    canvas.paste(_c22, (1274, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [1274, 1, 1314, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/23_icon_Oet_tickets_275PARKBKCOM.png
try:
    _c23 = get_crop(23, 1344, 898)
    canvas.paste(_c23, (48, 1918), _c23)
except Exception:
    pass
layout["Oet_tickets_+_275PARKBKCO"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/24_icon_Show_8PM.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Show_8PM"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/25_icon_Ticket_sales_end_soon.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (288, 2804), _c25)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/26_icon_Show_8PM.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["Show_8PM"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/27_icon_Ticket_sales_end_soon.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/28_text_9.25.png
try:
    _c28 = get_crop(28, 94, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["9.25"] = [20, 15, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/29_text_4_140_events.png
try:
    _c29 = get_crop(29, 372, 135)
    canvas.paste(_c29, (54, 390), _c29)
except Exception:
    pass
layout["4,140_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/30_text_Von.png
try:
    _c30 = get_crop(30, 77, 43)
    canvas.paste(_c30, (94, 1708), _c30)
except Exception:
    pass
layout["Von"] = [94, 1708, 171, 1751]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/31_text_Promoted.png
try:
    _c31 = get_crop(31, 193, 43)
    canvas.paste(_c31, (94, 1777), _c31)
except Exception:
    pass
layout["Promoted"] = [94, 1777, 287, 1820]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_05_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-7/32_text_275PARK.png
try:
    _c32 = get_crop(32, 1344, 898)
    canvas.paste(_c32, (48, 1918), _c32)
except Exception:
    pass
layout["275PARK"] = [48, 1918, 1392, 2816]
