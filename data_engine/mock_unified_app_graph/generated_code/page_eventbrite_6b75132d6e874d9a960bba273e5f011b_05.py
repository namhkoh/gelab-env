# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_05
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7.png
# step_index: 5/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & UI structure drawing for the provided canvas (1440x2960)
# Uses provided variables: canvas (PIL.Image), draw (PIL.ImageDraw)

# Colors
bg_color = (255, 255, 255)           # main background (white)
status_bar_color = (189, 189, 189)   # top status bar grey
underline_blue = (44, 84, 255)       # bright blue for search underline
soft_circle = (231, 243, 255)        # very light blue for icon backgrounds
card_bg = (250, 250, 252)            # off-white card background for list area
divider_color = (236, 238, 242)      # subtle dividers

W, H = canvas.size

# 1) Fill background (canvas starts white, but enforce)
draw.rectangle([0, 0, W, H], fill=bg_color)

# 2) Status bar area (top)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# 3) Header underline (search input underline) - full width inset
underline_y = 360
draw.line((48, underline_y, W - 48, underline_y), fill=underline_blue, width=5)

# 4) Two circular light-blue icon backgrounds (for the "Nearby" and "Online events" groups).
#    These are only decorative backgrounds; actual icons/text will be pasted on top.
circle_radius = 54
left_center = (150, 450)
right_center = (630, 450)
draw.ellipse([left_center[0] - circle_radius, left_center[1] - circle_radius,
              left_center[0] + circle_radius, left_center[1] + circle_radius],
             fill=soft_circle, outline=None)
draw.ellipse([right_center[0] - circle_radius, right_center[1] - circle_radius,
              right_center[0] + circle_radius, right_center[1] + circle_radius],
             fill=soft_circle, outline=None)

# 5) Subtle divider between header/options area and the list of found locations
divider_y = 720
draw.line((24, divider_y, W - 24, divider_y), fill=divider_color, width=1)

# 6) Rounded card background behind the list of locations
list_top = 820
list_bottom = H - 80
card_margin = 24
card_bbox = [card_margin, list_top, W - card_margin, list_bottom]
# rounded_rectangle might not be available in very old PIL, but should be OK here
try:
    draw.rounded_rectangle(card_bbox, radius=12, fill=card_bg, outline=None)
except Exception:
    # fallback to normal rectangle if rounded not available
    draw.rectangle(card_bbox, fill=card_bg, outline=None)

# 7) Separator lines between each list item (use detected row y positions)
row_tops = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
for y in row_tops:
    # draw a faint line across the card area (inset to match card margins)
    draw.line((card_margin + 12, y, W - card_margin - 12, y), fill=divider_color, width=1)

# 8) Short subtle left accent (vertical) to visually separate header from list (decorative)
accent_x = 48
draw.line((accent_x, divider_y + 18, accent_x, divider_y + 62), fill=underline_blue, width=4)

# 9) Bottom area subtle fade bar to anchor the page (very light)
fade_h = 60
fade_top = H - fade_h
draw.rectangle([0, fade_top, W, H], fill=(255, 255, 255))

# End of background & structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/02_icon_8.11.png
try:
    _c2 = get_crop(2, 62, 65)
    canvas.paste(_c2, (112, 1), _c2)
except Exception:
    pass
layout["8.11"] = [112, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/03_icon_8.11.png
try:
    _c3 = get_crop(3, 61, 63)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["8.11"] = [179, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 62, 62)
    canvas.paste(_c4, (309, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [309, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/05_icon_8.11.png
try:
    _c5 = get_crop(5, 168, 168)
    canvas.paste(_c5, (0, 72), _c5)
except Exception:
    pass
layout["8.11"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 58)
    canvas.paste(_c6, (248, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [248, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 62)
    canvas.paste(_c7, (1320, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 1, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 85, 97)
    canvas.paste(_c8, (1310, 285), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 285, 1395, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/09_icon_San_Francisco.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 840), _c9)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/10_icon_District_of_Columbia.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1740), _c10)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/13_icon_Miami.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1200), _c13)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/14_icon_United_Kingdom.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 2100), _c14)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/15_icon_District_of_Columbia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1560), _c15)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/16_icon_Philadelphia.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1920), _c16)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/17_icon_District_of_Columbia.png
try:
    _c17 = get_crop(17, 1440, 132)
    canvas.paste(_c17, (0, 1560), _c17)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 53, 65)
    canvas.paste(_c18, (382, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/19_icon_8.11.png
try:
    _c19 = get_crop(19, 100, 65)
    canvas.paste(_c19, (9, 0), _c19)
except Exception:
    pass
layout["8.11"] = [9, 0, 109, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/20_text_San_Francisco.png
try:
    _c20 = get_crop(20, 1344, 129)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["San_Francisco"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/21_text_Nearby.png
try:
    _c21 = get_crop(21, 415, 114)
    canvas.paste(_c21, (48, 465), _c21)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/22_text_Online_events.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/23_text_Current_location.png
try:
    _c23 = get_crop(23, 415, 114)
    canvas.paste(_c23, (48, 465), _c23)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/24_text_Virtual_attendance.png
try:
    _c24 = get_crop(24, 452, 114)
    canvas.paste(_c24, (511, 465), _c24)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/25_text_Found_locations.png
try:
    _c25 = get_crop(25, 311, 50)
    canvas.paste(_c25, (44, 740), _c25)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 212, 55)
    canvas.paste(_c26, (44, 2288), _c26)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/27_text_New_York.png
try:
    _c27 = get_crop(27, 154, 38)
    canvas.paste(_c27, (47, 2353), _c27)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/28_text_Atlanta.png
try:
    _c28 = get_crop(28, 163, 52)
    canvas.paste(_c28, (44, 2468), _c28)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/29_text_Georgia.png
try:
    _c29 = get_crop(29, 133, 43)
    canvas.paste(_c29, (45, 2533), _c29)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/30_clickable_New_York.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2280), _c30)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_05_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-7/31_clickable_Atlanta.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2460), _c31)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
