# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_05
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-7.png
# step_index: 5/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 72)], fill="#bfbfbf")

# Main background (slightly warm white to match screenshot)
draw.rectangle([(0, 72), (1440, 2960)], fill="#ffffff")

# Header underline (blue) below the title area
# Title area sits roughly between y=120 and y=320; place underline at ~y=336
underline_y = 336
draw.line([(40, underline_y), (1400, underline_y)], fill="#2B57FF", width=6)

# Subtle header bottom divider shadow
draw.line([(40, underline_y+9), (1400, underline_y+9)], fill="#e9ecff", width=1)

# Light rounded card behind the "Nearby" / "Online events" area
card_box = (28, 380, 1412, 620)
draw.rounded_rectangle(card_box, radius=16, fill="#ffffff", outline=None)

# Soft shadow under that card
shadow_y0 = 620
draw.rectangle([(28, shadow_y0), (1412, shadow_y0+6)], fill="#f5f6fa")

# Circular badges behind the two top options (only backgrounds, icons/texts will be pasted on top)
# Left badge (Nearby / Current location)
left_center = (160, 520)
left_radius = 56
draw.ellipse([(left_center[0]-left_radius, left_center[1]-left_radius),
              (left_center[0]+left_radius, left_center[1]+left_radius)],
             fill="#eaf2ff")

# Inner lighter ring for left badge
inner_lr = 34
draw.ellipse([(left_center[0]-inner_lr, left_center[1]-inner_lr),
              (left_center[0]+inner_lr, left_center[1]+inner_lr)],
             fill="#d4e6ff")

# Right badge (Online events / Virtual attendance)
right_center = (720, 520)
right_radius = 56
draw.ellipse([(right_center[0]-right_radius, right_center[1]-right_radius),
              (right_center[0]+right_radius, right_center[1]+right_radius)],
             fill="#eaf2ff")

inner_rr = 34
draw.ellipse([(right_center[0]-inner_rr, right_center[1]-inner_rr),
              (right_center[0]+inner_rr, right_center[1]+inner_rr)],
             fill="#d4e6ff")

# Thin divider line separating the header area from the list content
draw.line([(28, 720), (1412, 720)], fill="#eef0f5", width=1)

# Found locations title area subtle background (do not draw text)
draw.rectangle([(28, 720), (1412, 780)], fill="#ffffff")

# Draw separators for the list items (rows), aligned with detected row heights (each ~132px)
row_start_y = 840
row_height = 132
num_rows = 12  # draw a generous number of separators down the page
for i in range(num_rows + 1):
    y = row_start_y + i * row_height
    if y <= 2960:
        # full-width subtle separator
        draw.line([(40, y), (1400, y)], fill="#f1f2f6", width=1)

# Subtle left padding guide line (very faint) to emphasize content column (not text)
draw.line([(40, 72), (40, 2960)], fill="#ffffff", width=1)

# Light right padding guide line (very faint)
draw.line([(1400, 72), (1400, 2960)], fill="#ffffff", width=1)

# A gentle bottom area gradient suggestion (simple rectangle to hint content end)
draw.rectangle([(0, 2860), (1440, 2960)], fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 68)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 65)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/02_icon_7.28.png
try:
    _c2 = get_crop(2, 62, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.28"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/03_icon_7.28.png
try:
    _c3 = get_crop(3, 61, 64)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["7.28"] = [114, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 63, 62)
    canvas.paste(_c4, (308, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/05_icon_7.28.png
try:
    _c5 = get_crop(5, 168, 168)
    canvas.paste(_c5, (0, 72), _c5)
except Exception:
    pass
layout["7.28"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 49, 58)
    canvas.paste(_c6, (249, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 62)
    canvas.paste(_c7, (1320, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 1, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 85, 96)
    canvas.paste(_c8, (1310, 286), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 286, 1395, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/09_icon_District_of_Columbia.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1740), _c9)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/10_icon_San_Francisco.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 840), _c10)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/13_icon_United_Kingdom.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 2100), _c13)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/14_icon_District_of_Columbia.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1560), _c14)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/15_icon_Philadelphia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1920), _c15)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/16_icon_Miami.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1200), _c16)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/17_icon_7.28.png
try:
    _c17 = get_crop(17, 93, 64)
    canvas.paste(_c17, (14, 1), _c17)
except Exception:
    pass
layout["7.28"] = [14, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 54, 66)
    canvas.paste(_c18, (382, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 0, 436, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/19_text_New_York.png
try:
    _c19 = get_crop(19, 1344, 129)
    canvas.paste(_c19, (48, 264), _c19)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/20_text_Nearby.png
try:
    _c20 = get_crop(20, 415, 114)
    canvas.paste(_c20, (48, 465), _c20)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/22_text_Current_location.png
try:
    _c22 = get_crop(22, 415, 114)
    canvas.paste(_c22, (48, 465), _c22)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/23_text_Virtual_attendance.png
try:
    _c23 = get_crop(23, 452, 114)
    canvas.paste(_c23, (511, 465), _c23)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/24_text_Found_locations.png
try:
    _c24 = get_crop(24, 311, 50)
    canvas.paste(_c24, (44, 740), _c24)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/29_clickable_New_York.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2280), _c29)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_05_2024_4_23_19_27_45f56b06f31541079045047b6d542613-7/30_clickable_Atlanta.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2460), _c30)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
