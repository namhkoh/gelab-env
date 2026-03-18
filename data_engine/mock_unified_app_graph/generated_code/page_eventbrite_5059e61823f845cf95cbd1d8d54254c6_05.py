# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_05
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7.png
# step_index: 5/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background structure for the UI using provided `canvas` and `draw`.
# Available: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts.

# Canvas baseline (should already be white) - ensure full white background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top)
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill="#F3F4F6")
# subtle bottom edge/shadow of status bar
draw.line([(0, STATUS_H), (1440, STATUS_H)], fill="#E6E8EB", width=1)

# Top toolbar area (below status bar) - keep it light/clean
TOOLBAR_TOP = STATUS_H
TOOLBAR_BOTTOM = 200
draw.rectangle([(0, TOOLBAR_TOP), (1440, TOOLBAR_BOTTOM)], fill="#FFFFFF")
# faint divider under toolbar
draw.line([(48, TOOLBAR_BOTTOM), (1392, TOOLBAR_BOTTOM)], fill="#E8ECF2", width=2)

# Blue underline under the page title (full width with horizontal margins)
# Place it where the title area would be (not drawing the title text itself)
UNDERLINE_Y = 336
draw.line([(48, UNDERLINE_Y), (1392, UNDERLINE_Y)], fill="#2A56FF", width=4)

# Group background for the two "chips" (Nearby / Online events) area
# A very subtle rounded rectangle to indicate a grouping/background
CHIPS_TOP = UNDERLINE_Y + 36
CHIPS_BOTTOM = CHIPS_TOP + 180
draw.rounded_rectangle([(36, CHIPS_TOP), (1404, CHIPS_BOTTOM)], radius=18, fill="#FBFDFF", outline=None)

# Add a very faint horizontal separator to mark the "Found locations" section start
FOUND_DIV_Y = 720
draw.line([(48, FOUND_DIV_Y), (1392, FOUND_DIV_Y)], fill="#F0F2F5", width=1)

# Draw subtle separators between list rows (visual structure only)
# The list items start at ~840 and each detected item uses ~132px height.
list_start = 840
item_h = 132
# Draw separators for several rows but avoid drawing inside the last two clickable items zone (y >= 2280)
for i in range(0, 12):
    y = list_start + (i + 1) * item_h
    if y >= 2280 and y <= 2592:
        # skip separators that fall inside the detected full-width clickable areas near the bottom
        continue
    if y < 2960:
        draw.line([(48, y), (1392, y)], fill="#F4F6F8", width=1)

# Right-side subtle vertical margin guideline (visual structure)
draw.line([(48, 0), (48, 2960)], fill="#FFFFFF", width=0)  # noop white (keeps margin concept without visible duplication)

# Bottom area (just ensure full-bleed white remains consistent)
draw.rectangle([(0, 2592), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 68)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 65)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/02_icon_7.34.png
try:
    _c2 = get_crop(2, 62, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.34"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/03_icon_7.34.png
try:
    _c3 = get_crop(3, 60, 63)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["7.34"] = [114, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/04_icon_7.34.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 62)
    canvas.paste(_c5, (308, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 49, 57)
    canvas.paste(_c6, (249, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 5, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 63)
    canvas.paste(_c7, (1320, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 0, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 85, 96)
    canvas.paste(_c8, (1310, 286), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 286, 1395, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/09_icon_District_of_Columbia.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1740), _c9)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/10_icon_7.34.png
try:
    _c10 = get_crop(10, 94, 64)
    canvas.paste(_c10, (14, 0), _c10)
except Exception:
    pass
layout["7.34"] = [14, 0, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/11_icon_San_Francisco.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 840), _c11)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/12_icon_Chicago.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1380), _c12)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/13_icon_Los_Angeles.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1020), _c13)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/14_icon_United_Kingdom.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 2100), _c14)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/15_icon_District_of_Columbia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1560), _c15)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/16_icon_Miami.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1200), _c16)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/17_icon_Philadelphia.png
try:
    _c17 = get_crop(17, 1440, 132)
    canvas.paste(_c17, (0, 1920), _c17)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 53, 66)
    canvas.paste(_c18, (382, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 0, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/19_text_Los_Angeles.png
try:
    _c19 = get_crop(19, 1344, 129)
    canvas.paste(_c19, (48, 264), _c19)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/20_text_Nearby.png
try:
    _c20 = get_crop(20, 415, 114)
    canvas.paste(_c20, (48, 465), _c20)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/22_text_Current_location.png
try:
    _c22 = get_crop(22, 415, 114)
    canvas.paste(_c22, (48, 465), _c22)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/23_text_Virtual_attendance.png
try:
    _c23 = get_crop(23, 452, 114)
    canvas.paste(_c23, (511, 465), _c23)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/24_text_Found_locations.png
try:
    _c24 = get_crop(24, 311, 50)
    canvas.paste(_c24, (44, 740), _c24)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/29_clickable_New_York.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2280), _c29)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_05_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-7/30_clickable_Atlanta.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2460), _c30)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
